//! Symbolic memory model for the VM.

#[cfg(rapx_const_ext)]
use rustc_middle::ty::consts::ConstExt;
use rustc_middle::{
    mir::{Local, Place, ProjectionElem},
    ty::{Ty, TyKind},
};
use z3::ast::{Ast, Int};

use super::state::{AllocId, AllocKind, Allocation, ByteInfo, Provenance, ValueInvariants, ValueSource, VmState, VmValue};

impl<'z3, 'tcx> VmState<'z3, 'tcx> {
    pub(crate) fn address_of_place(&mut self, place: &Place<'tcx>) -> Option<VmValue<'z3, 'tcx>> {
        self.ensure_local_allocation(place.local);

        let zero = Int::from_u64(self.z3_ctx, 0);

        if place.projection.is_empty() {
            let base_addr = self.local_address(place.local);
            let ty = self.body().local_decls[place.local].ty;
            // Prefer the local's value provenance over the stack-allocation
            // provenance. For Box/Vec parameters, the value tracks the heap
            // allocation while slots tracks the stack location.
            let provenance = self
                .current_frame.local_values
                .get(&place.local)
                .and_then(|v| v.provenance.clone())
                .or_else(|| {
                    self.current_frame.local_alloc
                        .get(&place.local)
                        .copied()
                        .map(|alloc_id| Provenance {
                            alloc_id,
                            offset: zero,
                            offset_kind: None,
                        })
                });
            return Some(VmValue {
                z3_term: base_addr,
                ty,
                provenance,
                invariants: ValueInvariants::default(),
                source: ValueSource::None,
            });
        }

        let mut term = self.local_address(place.local);
        let mut provenance: Option<Provenance<'z3>> = self
            .current_frame.local_alloc
            .get(&place.local)
            .copied()
            .map(|alloc_id| Provenance {
                alloc_id,
                offset: zero.clone(),
                offset_kind: None,
            });
        let mut current_ty = self.body().local_decls[place.local].ty;
        let mut field_path: Vec<usize> = Vec::new();
        let mut view_ty = current_ty;

        for proj in place.projection.iter() {
            let mut handled = false;
            if let ProjectionElem::Index(local) = proj {
                // The element stride is the *element* size, not the container
                // size: peel `[T; N]` / `[T]` down to `T` (mirroring state.rs's
                // `Index` arm).  Keep `max(1)` so the stride matches the array
                // allocation's `elem_size` (`N·max(1)`), letting the SMT cancel
                // the factor for `idx + 1 <= N`.
                let elem_ty = match current_ty.kind() {
                    TyKind::Array(e, _) | TyKind::Slice(e) => *e,
                    _ => current_ty,
                };
                let elem_sz = Int::from_u64(self.z3_ctx, self.size_of_ty(elem_ty).max(1));
                if let Some(val) = self.current_frame.local_values.get(&local) {
                    if let Some(idx) = val.z3_term.simplify().as_u64() {
                        let scaled = Int::mul(self.z3_ctx, &[&Int::from_u64(self.z3_ctx, idx), &elem_sz]);
                        term = Int::add(self.z3_ctx, &[&term, &scaled]);
                        if let Some(ref mut prov) = provenance {
                            prov.offset = Int::add(self.z3_ctx, &[&prov.offset, &scaled]);
                        }
                        handled = true;
                    }
                }
                if !handled {
                    let idx = self.fresh_int("idx");
                    let scaled = Int::mul(self.z3_ctx, &[&idx, &elem_sz]);
                    term = Int::add(self.z3_ctx, &[&term, &scaled]);
                    if let Some(ref mut prov) = provenance {
                        prov.offset = Int::add(self.z3_ctx, &[&prov.offset, &scaled]);
                    }
                }
                continue;
            }
            match proj.kind() {
                ProjectionElem::Field(field_idx, _) => {
                    let fidx = field_idx.as_usize();
                    field_path.push(fidx);
                    let field_offset = self.field_offset_in_bytes(current_ty, fidx);
                    let field_off = Int::from_u64(self.z3_ctx, field_offset);
                    // Advance `current_ty` to the field's type so that subsequent
                    // projections resolve their offsets against the right layout.
                    let field_ty = match current_ty.kind() {
                        TyKind::Adt(adt_def, substs) => {
                            let variant = adt_def.non_enum_variant();
                            variant
                                .fields
                                .get(rustc_abi::FieldIdx::from_usize(fidx))
                                .map(|f| crate::helpers::mir_utils::field_ty(self.tcx, f, substs))
                                .unwrap_or(current_ty)
                        }
                        _ => current_ty,
                    };
                    // A DST slice field (e.g. `CStr { inner: [u8] }`) carries its
                    // own allocation so its length stays symbolic.  Prefer that
                    // allocation over the parent struct's byte-offset address,
                    // otherwise `&raw const self.inner` collapses the slice length
                    // to the struct's own (minimal) size.
                    let field_replacement = match field_ty.kind() {
                        TyKind::Slice(_) => self
                            .field_value(place.local, &field_path)
                            .and_then(|fv| fv.provenance.clone().map(|p| (fv.z3_term.clone(), p))),
                        TyKind::Array(..) => {
                            // An array field decomposed into its own allocation by
                            // `decompose_pointee_fields` (e.g. `keys: [MaybeUninit<K>; N]`)
                            // carries a concrete element count. Prefer that allocation
                            // over the parent struct's byte-offset address so
                            // `&(*leaf).keys` keeps `len = N` for downstream InBound.
                            let alloc = provenance.as_ref().map(|p| p.alloc_id);
                            alloc.and_then(|a| {
                                self.memory.fields
                                    .get(&(a, view_ty, field_path.clone()))
                                    .and_then(|fv| {
                                        fv.provenance.clone().map(|p| (fv.z3_term.clone(), p))
                                    })
                            })
                        }
                        _ => None,
                    };
                    match field_replacement {
                        Some((fv_term, fv_prov)) => {
                            term = fv_term;
                            provenance = Some(fv_prov);
                        }
                        None => {
                            term = Int::add(self.z3_ctx, &[&term, &field_off]);
                            if let Some(ref mut prov) = provenance {
                                prov.offset = Int::add(self.z3_ctx, &[&prov.offset, &field_off]);
                            }
                        }
                    }
                    current_ty = field_ty;
                }
                ProjectionElem::Deref => {
                    field_path.clear();
                    let pointed = self.current_frame.local_values.get(&place.local)?;
                    term = pointed.z3_term.clone();
                    provenance = pointed.provenance.clone();
                    // For fat pointers (aggregates without provenance),
                    // use the first field's provenance (the data pointer).
                    if provenance.is_none() && matches!(pointed.ty.kind(), TyKind::RawPtr(..)) {
                        if let Some(field0) = self.field_value(place.local, &[0]) {
                            provenance = field0.provenance.clone();
                        }
                    }
                    if let TyKind::Ref(_, deref_ty, _) = current_ty.kind() {
                        current_ty = *deref_ty;
                    } else if let TyKind::RawPtr(deref_ty, _) = current_ty.kind() {
                        current_ty = *deref_ty;
                    }
                    view_ty = current_ty;
                }
                _ => {
                    return None;
                }
            }
        }

        let ty = place.ty(self.body(), self.tcx).ty;
        Some(VmValue {
            z3_term: term,
            ty,
            provenance,
            invariants: ValueInvariants::default(),
            source: ValueSource::None,
        })
    }

    /// Lazily create a stack allocation for a MIR local if one doesn't exist.
    pub(crate) fn ensure_local_allocation(&mut self, local: Local) {
        if self.current_frame.local_alloc.contains_key(&local) {
            return;
        }
        let ty = self.body().local_decls[local].ty;
        let align = self.align_sym(ty);
        // Generate the base address symbol directly (the address lives only in
        // `Allocation::base` now; `local_address` reads it back from there).
        let name = format!("addr__{}", local.as_usize());
        let base = Int::new_const(self.z3_ctx, name.as_str());
        let id = AllocId(self.memory.allocations.len());
        // For arrays, track the element type (not the array type) so that
        // len() computes `size / elem_size` correctly.  When the element size
        // is unknown (a generic `T`), `size_of::<[T; N]>()` collapses to 0, so
        // instead record the element count `N` as a symbolic term — this keeps
        // `len() = size / elem_size` equal to `N`, letting downstream
        // InBound checks (e.g. `get_unchecked_mut(idx)` where `idx < N`) be
        // discharged against the loop's `idx < N` path condition.
        let (size_term, element_ty, slice_len) = match ty.kind() {
            TyKind::Array(elem, const_len) => {
                // Concrete element size (`.max(1)` so a generic `T` collapses to
                // 1 byte, keeping `len() = size / elem_size` equal to the
                // symbolic element count `N`).  Deliberately *not* the symbolic
                // `size_sym(elem)`: materializing `sizeof_MaybeUninit<T>` here
                // would let `access_bytes` read it back as an unbounded access
                // size, breaking `Allocated(&mut MaybeUninit<T>, T, 1)` against
                // the iterator provenance (array_try_from_fn_ext).
                let elem_size = self.size_of_ty(*elem).max(1);
                let n_term = self.const_len_term(const_len);
                let size = match n_term.as_u64() {
                    Some(n) => Int::from_u64(self.z3_ctx, n.saturating_mul(elem_size)),
                    None => Int::mul(self.z3_ctx, &[&n_term, &Int::from_u64(self.z3_ctx, elem_size)]),
                };
                (size, Some(*elem), Some(n_term))
            }
            _ => {
                let size = self
                    .struct_size_sym(ty)
                    .unwrap_or_else(|| self.size_sym(ty));
                (size, Some(ty), None)
            }
        };
        let mut alloc = Allocation::new(base, size_term, align, element_ty, AllocKind::Object);
        if let Some(len) = slice_len {
            alloc.set_slice_len(len);
        }
        self.memory.allocations.push(alloc);
        self.current_frame.local_alloc.insert(local, id);
    }

    pub(crate) fn field_offset_in_bytes(&self, ty: Ty<'tcx>, field_idx: usize) -> u64 {
        crate::helpers::mir_utils::field_offset_in_bytes(
            self.tcx,
            self.current_frame.current_def_id,
            ty,
            field_idx,
        )
    }

    pub(crate) fn size_of_ty(&self, ty: Ty<'tcx>) -> u64 {
        crate::helpers::mir_utils::layout_of_ty(self.tcx, self.current_frame.current_def_id, ty)
            .map(|l| l.size.bytes())
            .unwrap_or(0)
    }

    pub(crate) fn align_of_ty(&self, ty: Ty<'tcx>) -> u64 {
        crate::helpers::mir_utils::layout_of_ty(self.tcx, self.current_frame.current_def_id, ty)
            .map(|l| l.align.abi.bytes())
            .unwrap_or(1)
    }

    pub(crate) fn allocation_size(&self, alloc_id: AllocId) -> &Int<'z3> {
        &self.alloc(alloc_id).size
    }

    pub(crate) fn allocation_base(&self, alloc_id: AllocId) -> &Int<'z3> {
        &self.alloc(alloc_id).base
    }

    /// Get the element size (in bytes) for a pointer type, peeling
    /// through `*const T`, `*mut T`, `&T`, and `&[T]` to find `size_of(T)`.
    pub(crate) fn pointee_elem_size(&self, ty: Ty<'tcx>) -> u64 {
        let inner = match ty.kind() {
            TyKind::RawPtr(inner_ty, _) | TyKind::Ref(_, inner_ty, _) => *inner_ty,
            _ => ty,
        };
        match inner.kind() {
            TyKind::Slice(elem) => self.size_of_ty(*elem),
            _ => self.size_of_ty(inner),
        }
    }

    /// Element size of `ty` as a symbolic Z3 term.  For concrete types this is
    /// the constant byte size; for a generic type whose `size_of` is unknown
    /// (an unconstrained `T`) it is a single reusable symbolic constant with
    /// `>= 0` (so `T` may be a ZST).  Using the same constant everywhere (ptr
    /// strides, access counts, allocation sizes) lets SMT cancel the factor in
    /// `InBound`.
    pub(crate) fn size_sym(&mut self, ty: Ty<'tcx>) -> Int<'z3> {
        let ty = peel_slice_elem(ty);
        let size = self.size_of_ty(ty);
        if size > 0 || !crate::helpers::mir_utils::ty_has_type_param(ty) {
            return Int::from_u64(self.z3_ctx, size);
        }
        if let Some(s) = self.constraints.term_caches.sizes.get(&ty) {
            return s.clone();
        }
        let s = self.fresh_int(&format!("sizeof_{ty}"));
        self.constraints.term_caches.sizes.insert(ty, s.clone());
        let zero = Int::from_u64(self.z3_ctx, 0);
        self.constraints.assertions.push(s.ge(&zero));
        s
    }

    /// The array length `N` as a Z3 term (concrete value or symbolic const
    /// generic).  The symbolic name mirrors `value_of_operand`'s formatting so it
    /// is *identical* to the `const N` term appearing in path conditions.
    fn const_len_term(&self, const_len: &rustc_middle::ty::Const<'tcx>) -> Int<'z3> {
        match const_len.try_to_target_usize(self.tcx) {
            Some(v) => Int::from_u64(self.z3_ctx, v),
            None => {
                let const_text = format!("Ty({:?}, {:?})", self.tcx.types.usize, const_len);
                let name = format!("const_{}", const_text.replace([':', '#', ' '], "_"));
                Int::new_const(self.z3_ctx, name.as_str())
            }
        }
    }

    /// Read-only sibling of [`size_sym`](Self::size_sym): returns the symbolic
    /// size for `ty`, falling back to `1` when the symbolic constant has not
    /// been created yet (e.g. a checker invoked before the exec phase created
    /// it).  Non-ZST concrete types return their constant byte size; a concrete
    /// ZST or a not-yet-created generic constant falls back to `1` — a non-zero
    /// element size keeps `size / elem_size` derivations from dividing by zero.
    pub(crate) fn size_sym_read(&self, ty: Ty<'tcx>) -> Int<'z3> {
        let ty = peel_slice_elem(ty);
        let size = self.size_of_ty(ty);
        if size > 0 {
            return Int::from_u64(self.z3_ctx, size);
        }
        self.constraints.term_caches.sizes
            .get(&ty)
            .cloned()
            .unwrap_or_else(|| Int::from_u64(self.z3_ctx, 1))
    }

    /// The *symbolic* element-size term of `alloc_id`'s element type, when it is
    /// a generic type parameter (so it may be `0` for a ZST or `≥ 1` for a
    /// non-ZST).  Returns `None` for concrete element types, where the size is a
    /// known constant and no case split is needed.
    pub(crate) fn generic_elem_size(&self, alloc_id: AllocId) -> Option<Int<'z3>> {
        let elem_ty = self.alloc(alloc_id).element_ty.as_ty()?;
        if !crate::helpers::mir_utils::ty_has_type_param(elem_ty) {
            return None;
        }
        let s = self.size_sym_read(elem_ty);
        if s.simplify().as_u64().is_some() {
            return None;
        }
        Some(s)
    }

    /// Alignment of `ty` as a symbolic Z3 term.  For a concrete type this is
    /// the constant byte alignment; for a generic type it is a reusable
    /// symbolic constant `align_T` with `>= 1`, lower-bounded by the trait
    /// bounds' minimum alignment, and linked to the element size by the layout
    /// constraint `sizeof_T % align_T == 0` (a type's size is always a multiple
    /// of its alignment).  For a generic struct, its alignment is additionally
    /// constrained to be a multiple of each field's alignment, so a field
    /// pointer (`(*node).value`) inherits the container's alignment.
    pub(crate) fn align_sym(&mut self, ty: Ty<'tcx>) -> Int<'z3> {
        let ty = peel_slice_elem(ty);
        // An array's alignment equals its element's alignment.
        if let TyKind::Array(elem, _) = ty.kind() {
            return self.align_sym(*elem);
        }
        let align = self.align_of_ty(ty);
        if align > 1 || !crate::helpers::mir_utils::ty_has_type_param(ty) {
            return Int::from_u64(self.z3_ctx, align);
        }
        if let Some(a) = self.constraints.term_caches.aligns.get(&ty) {
            return a.clone();
        }
        let a = self.fresh_int(&format!("align_{ty}"));
        self.constraints.term_caches.aligns.insert(ty, a.clone());
        let one = Int::from_u64(self.z3_ctx, 1);
        let zero = Int::from_u64(self.z3_ctx, 0);
        self.constraints.assertions.push(a.ge(&one));
        // Lower bound from the trait bounds (0 for an unconstrained `T`): any
        // implementor is at least this aligned.
        let min_a =
            crate::helpers::mir_utils::min_align_of_generic_param(self.tcx, self.current_frame.current_def_id, ty);
        if min_a > 1 {
            self.constraints.assertions
                .push(a.ge(&Int::from_u64(self.z3_ctx, min_a)));
        }
        // Upper bound from the trait bounds (0 for an unconstrained `T`): any
        // implementor is at most this aligned, which is what lets a cross-cast
        // from a *more* aligned source (`&[U]` -> `*const T`) be discharged.
        let max_a =
            crate::helpers::mir_utils::max_align_of_generic_param(self.tcx, self.current_frame.current_def_id, ty);
        if max_a > 0 {
            self.constraints.assertions
                .push(a.le(&Int::from_u64(self.z3_ctx, max_a)));
        }
        // A struct's alignment is a multiple of each field's alignment (both
        // are powers of two).  Pointer fields have a *concrete* alignment, so
        // this terminates even for recursively-defined containers.
        if let TyKind::Adt(adt_def, substs) = ty.kind() {
            if !adt_def.is_enum() {
                let variant = adt_def.non_enum_variant();
                for field in variant.fields.iter() {
                    let field_ty = crate::helpers::mir_utils::field_ty(self.tcx, field, substs);
                    let field_align = self.align_sym(field_ty);
                    self.constraints.assertions.push(a.rem(&field_align)._eq(&zero));
                }
                // A struct's size is at least the sum of its fields (padding may
                // add more).  This relates the symbolic `sizeof_Struct` constant
                // to the field-sum that `size_of::<Struct>()` lowers to (e.g.
                // `SIZE<LeafNode> + SIZE<[...; 12]>` for `InternalNode`), so
                // `Allocated(p, u8, layout.size)` is discharged against the
                // allocation size.
                if let Some(sum) = self.struct_size_sym(ty) {
                    let size = self.size_sym(ty);
                    self.constraints.assertions.push(size.ge(&sum));
                }
            }
        }
        // Layout invariant: a type's size is a multiple of its alignment.
        let size = self.size_sym(ty);
        self.constraints.assertions.push(size.rem(&a)._eq(&zero));
        a
    }

    /// Read-only sibling of [`align_sym`](Self::align_sym): returns the
    /// symbolic alignment for `ty`, falling back to the trait bounds' minimum
    /// alignment when the constant has not been created yet (e.g. a generic `U`
    /// that only appears in a cast/contract, never as an allocation element
    /// type).  Concrete types return their constant alignment.
    pub(crate) fn align_sym_read(&self, ty: Ty<'tcx>) -> Int<'z3> {
        let ty = peel_slice_elem(ty);
        // An array's alignment equals its element's alignment.
        if let TyKind::Array(elem, _) = ty.kind() {
            return self.align_sym_read(*elem);
        }
        let align = self.align_of_ty(ty);
        if align > 1 {
            return Int::from_u64(self.z3_ctx, align);
        }
        if let Some(a) = self.constraints.term_caches.aligns.get(&ty) {
            return a.clone();
        }
        let min_a =
            crate::helpers::mir_utils::min_align_of_generic_param(self.tcx, self.current_frame.current_def_id, ty);
        Int::from_u64(self.z3_ctx, min_a.max(1))
    }

    /// Size of a struct/ADT as the *sum* of its fields' sizes (each via
    /// [`size_sym`](Self::size_sym)).  This lower-bounds the real layout so a
    /// field reference (`Allocated(&alloc)`) can be discharged against the
    /// struct allocation (`sizeof_A <= 8 + 8 + sizeof_A`).  Returns `None` for
    /// non-ADT or enum types.
    pub(crate) fn struct_size_sym(&mut self, ty: Ty<'tcx>) -> Option<Int<'z3>> {
        let TyKind::Adt(adt_def, substs) = ty.kind() else {
            return None;
        };
        if adt_def.is_enum() {
            return None;
        }
        let concrete = self.size_of_ty(ty);
        if concrete > 0 {
            return Some(Int::from_u64(self.z3_ctx, concrete));
        }
        let variant = adt_def.non_enum_variant();
        let mut total = Int::from_u64(self.z3_ctx, 0);
        for field in variant.fields.iter() {
            let field_ty = crate::helpers::mir_utils::field_ty(self.tcx, field, substs);
            let field_size = self
                .struct_size_sym(field_ty)
                .unwrap_or_else(|| self.size_sym(field_ty));
            total = Int::add(self.z3_ctx, &[&total, &field_size]);
        }
        Some(total)
    }

    // ── Per-byte state (`Memory::bytes`) ────────────────────────────────

    /// Record a per-byte symbolic value at a concrete offset in an allocation.
    pub(crate) fn record_byte_value(&mut self, alloc_id: AllocId, offset: usize, term: Int<'z3>) {
        let byte = self.memory.bytes.entry((alloc_id, offset)).or_default();
        byte.value = Some(term);
        byte.init = true;
    }

    /// Mark a byte as initialized without changing its value.
    pub(crate) fn mark_byte_init(&mut self, alloc_id: AllocId, offset: usize) {
        self.memory.bytes.entry((alloc_id, offset)).or_default().init = true;
    }

    /// Mark a byte as known NUL (0x00).
    pub(crate) fn mark_byte_nul(&mut self, alloc_id: AllocId, offset: usize) {
        self.memory.bytes.entry((alloc_id, offset)).or_default().nul = Some(true);
    }

    /// Mark a byte as known non-NUL (!= 0x00).
    pub(crate) fn mark_byte_non_nul(&mut self, alloc_id: AllocId, offset: usize) {
        self.memory.bytes.entry((alloc_id, offset)).or_default().nul = Some(false);
    }

    /// Look up a per-byte Z3 term for a concrete offset in an allocation.
    pub(crate) fn get_byte_value(&self, alloc_id: AllocId, offset: usize) -> Option<&Int<'z3>> {
        self.memory.bytes
            .get(&(alloc_id, offset))
            .and_then(|b| b.value.as_ref())
    }

    /// Check whether a byte at a concrete offset is known to be initialized.
    pub(crate) fn is_byte_init(&self, alloc_id: AllocId, offset: usize) -> bool {
        self.memory.bytes.get(&(alloc_id, offset)).is_some_and(|b| b.init)
    }

    /// Check whether a byte at a concrete offset is known to be NUL.
    pub(crate) fn is_byte_nul(&self, alloc_id: AllocId, offset: usize) -> bool {
        self.memory.bytes
            .get(&(alloc_id, offset))
            .is_some_and(|b| b.nul == Some(true))
    }

    /// Check whether a byte at a concrete offset is known to be non-NUL.
    pub(crate) fn is_byte_non_nul(&self, alloc_id: AllocId, offset: usize) -> bool {
        self.memory.bytes
            .get(&(alloc_id, offset))
            .is_some_and(|b| b.nul == Some(false))
    }

    /// Return all known (offset, term) pairs for an allocation, sorted by offset.
    pub(crate) fn alloc_byte_values(&self, alloc_id: AllocId) -> Vec<(usize, &Int<'z3>)> {
        let mut pairs: Vec<_> = self
            .memory.bytes
            .iter()
            .filter_map(|((aid, off), byte)| {
                if *aid == alloc_id {
                    byte.value.as_ref().map(|term| (*off, term))
                } else {
                    None
                }
            })
            .collect();
        pairs.sort_by_key(|(off, _)| *off);
        pairs
    }

    /// Collect all offsets known to be NUL in an allocation.
    pub(crate) fn alloc_nul_offsets(&self, alloc_id: AllocId) -> Vec<usize> {
        self.memory.bytes
            .iter()
            .filter_map(|((aid, off), byte)| {
                if *aid == alloc_id && byte.nul == Some(true) {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Collect all offsets known to be non-NUL in an allocation.
    pub(crate) fn alloc_non_nul_offsets(&self, alloc_id: AllocId) -> Vec<usize> {
        self.memory.bytes
            .iter()
            .filter_map(|((aid, off), byte)| {
                if *aid == alloc_id && byte.nul == Some(false) {
                    Some(*off)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Copy all per-byte tracking (value, init, NUL knowledge) from one
    /// allocation to another.
    pub(crate) fn copy_byte_tracking(&mut self, src: AllocId, src_offset: usize, dst: AllocId) {
        let infos: Vec<(usize, ByteInfo<'z3>)> = self
            .memory.bytes
            .iter()
            .filter(|((aid, _), _)| *aid == src)
            .map(|((_, off), byte)| (*off, byte.clone()))
            .collect();
        for (off, byte) in infos {
            // The destination allocation starts at `src_offset` into the source,
            // so shift each tracked byte by that offset (`src[off]` → `dst[off -
            // src_offset]`).  Bytes before `src_offset` lie outside the sub-slice
            // and are dropped.
            if off >= src_offset {
                self.memory.bytes.insert((dst, off - src_offset), byte);
            }
        }
    }

    // ── Per-allocation fields (`Memory::fields`) ─────────────────────────

    /// The value at a field offset *within an allocation* viewed as `view_ty`.
    ///
    /// This is the memory-contents layer ([`Memory::fields`]), the counterpart
    /// to [`Self::field_value`]'s binding-value layer ([`FrameState::field_values`]):
    /// `field_value` asks "what value does the aggregate bound to `local` hold
    /// at field `path`?", while `load_field` asks "what value sits at
    /// `base(alloc_id) + offset(path)` interpreted as `view_ty`?".  The viewed
    /// type is part of the key so reinterpret casts (`LeafNode` ↔
    /// `InternalNode`) resolve to the right field view.
    pub(crate) fn load_field(
        &self,
        alloc_id: AllocId,
        view_ty: Ty<'tcx>,
        path: &[usize],
    ) -> Option<&VmValue<'z3, 'tcx>> {
        self.memory.fields.get(&(alloc_id, view_ty, path.to_vec()))
    }
}

/// Peel a slice type `[T]` to its element `T` (other types unchanged).
fn peel_slice_elem(ty: Ty<'_>) -> Ty<'_> {
    match ty.kind() {
        TyKind::Slice(elem) => *elem,
        _ => ty,
    }
}
