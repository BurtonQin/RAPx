//! Symbolic VM state types.
//!
//! The data structures that represent the symbolic execution state: symbolic
//! values and their invariants, memory allocations, and the full execution
//! state at a program point.

use rustc_hir::def_id::DefId;
use rustc_middle::{
    mir::{Body, Local, Operand, Place, ProjectionElem},
    ty::{Region, Ty, TyCtxt},
};
use z3::{
    Context,
    ast::{Ast, Bool, Int},
};

use crate::compat::{FxHashMap, FxHashSet};
use crate::verify::{def_use::PlaceKey, path_extractor::Path};

/// Unique identifier for a heap or stack allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AllocId(pub usize);

/// The structure of a pointer's byte offset, when it has one.
///
/// Each variant lets the verifier use a cheaper, more precise proof: `Field`
/// carries the "field in-bounds" guarantee (`offset + size_of(field) <=
/// size_of(container)`, e.g. `Option::as_slice`), and `Element(k)` tracks the
/// element index so `InBound` checks `k + count <= slice_len` linearly instead
/// of the non-linear byte form `(k+count)·S <= len·S` (undecidable in Z3 NIA
/// for a generic `S`).
#[derive(Clone, Debug)]
pub(crate) enum OffsetKind<'ctx> {
    /// Compile-time field offset (`offset_of!`), including a first field at 0.
    Field,
    /// Element index from element-strided arithmetic (`ptr.add(k)`); the byte
    /// offset is `element · S`.
    Element(Int<'ctx>),
    /// A byte-strided or otherwise unclassifiable offset (`byte_add`).
    Byte,
}

/// Pointer provenance: which allocation, at what byte offset, and (when known)
/// what structure that offset has.
#[derive(Clone, Debug)]
pub(crate) struct Provenance<'ctx> {
    /// The allocation this pointer derives from.
    pub alloc_id: AllocId,
    /// Byte offset from the allocation base (always present). A freshly created
    /// pointer to the base of an allocation has `offset = 0`.
    pub offset: Int<'ctx>,
    /// The offset's structure. `None` means the pointer sits at the base
    /// (`offset == 0`) with no further structure; `Some(..)` records whether the
    /// offset is a compile-time field offset (`Field`), an element index from
    /// element-strided arithmetic (`Element`), or a byte-strided offset (`Byte`).
    pub offset_kind: Option<OffsetKind<'ctx>>,
}

/// Known invariants about a symbolic value.
///
/// `PartialEq`/`Eq` are deliberately *not* derived: `align_n` is a Z3 AST
/// whose equality is structural (`Z3_is_eq_ast`), not semantic, so comparing
/// two `ValueInvariants` would silently report semantically-equal values as
/// unequal.
#[derive(Clone, Debug, Default)]
pub(crate) struct ValueInvariants<'ctx> {
    pub non_null: bool,
    pub init: bool,
    pub in_bounds: bool,
    /// If Some(n), the value's term is known to satisfy `term % n == 0`.
    /// Set by alignment guards, Mul by power-of-two, and type alignment.
    /// `n` is a Z3 term so that a generic type's alignment (a symbolic
    /// `align_T`) can be carried the same way as a concrete alignment.
    pub align_n: Option<Int<'ctx>>,
}

/// A symbolic value tracked by the VM.
///
/// # Semantics of `term`
///
/// - For pointer/reference types (`&T`, `*const T`, `*mut T`, `Box<T>`, etc.):
///   `term` represents the **address** in the VM's logical address space.
/// - For scalar types (integers, `bool`, `char`): `term` represents the **value**.
/// - For aggregate types (struct, tuple, enum): `term` is the base address of
///   the stack allocation backing the aggregate.
///
/// When `provenance` is `Some`, the following relationship holds and is
/// asserted into the solver at check time:
///   `term == alloc[provenance.alloc_id].base + provenance.offset`
#[derive(Clone, Debug)]
pub(crate) struct VmValue<'ctx, 'tcx> {
    /// The Z3 integer term (address or scalar value, see struct docs).
    pub term: Int<'ctx>,
    /// Rust type, for layout queries.
    pub ty: Ty<'tcx>,
    /// Which allocation this pointer derives from and at what offset.
    pub provenance: Option<Provenance<'ctx>>,
    /// Known constraints on this value.
    pub invariants: ValueInvariants<'ctx>,
    /// Source tag: this scalar is a compile-time field offset (`offset_of!`).
    /// Propagated to a pointer's provenance when used as an `add`/`byte_add`
    /// offset.  It is origin information, not an SMT-assertable invariant, so it
    /// lives on `VmValue` rather than in `ValueInvariants`.
    pub field_offset: bool,
    /// Symbolic discriminant (variant index) for an ADT value whose variant is
    /// known symbolically (e.g. `Iterator::next` returns `Some(x) iff
    /// !is_empty`).  `None` for non-enum values or unknown variants.  Used by
    /// `Rvalue::Discriminant` so `switchInt` branches stay tied to the real
    /// condition instead of a fresh unconstrained symbol.
    pub discriminant: Option<Int<'ctx>>,
    /// Direct boolean condition for a comparison result (Le/Lt/Ge/Gt/Eq/Ne),
    /// recorded alongside the ite-encoded `term` so `switchInt`/`Assert` can
    /// emit a precise path condition (`offset <= len`) instead of
    /// `ite(cond, 1, 0) != 0`, which the SMT solver often fails to unfold.
    /// `None` for non-comparison values.
    pub bool_cond: Option<Bool<'ctx>>,
}

impl<'ctx, 'tcx> VmValue<'ctx, 'tcx> {
    pub(crate) fn new(term: Int<'ctx>, ty: Ty<'tcx>) -> Self {
        VmValue {
            term,
            ty,
            provenance: None,
            invariants: ValueInvariants::default(),
            field_offset: false,
            discriminant: None,
            bool_cond: None,
        }
    }

    /// Convenience: extract the `AllocId` from provenance, if any.
    pub(crate) fn provenance_alloc_id(&self) -> Option<AllocId> {
        self.provenance.as_ref().map(|p| p.alloc_id)
    }

    /// Symbolic enum discriminant, if known.
    pub(crate) fn discriminant(&self) -> Option<&Int<'ctx>> {
        self.discriminant.as_ref()
    }

    /// Direct boolean condition of a comparison result, if any.
    pub(crate) fn bool_cond(&self) -> Option<&Bool<'ctx>> {
        self.bool_cond.as_ref()
    }

    /// Whether this scalar is a compile-time `offset_of!` field offset.
    pub(crate) fn is_field_offset(&self) -> bool {
        self.field_offset
    }

    /// Whether this value is a pointer (carries provenance).
    pub(crate) fn is_pointer(&self) -> bool {
        self.provenance.is_some()
    }
}

/// The shape of an allocation. Replaces the implicit `slice_len` / `is_external`
/// combination that previously encoded the kind. `element_ty` (typed vs untyped)
/// and `parent`/`slice_data` (sub-view / slice-ref edges) stay separate fields.
#[derive(Clone, Debug)]
pub(crate) enum AllocKind<'ctx> {
    /// A single object: a `Box<T>` heap object, a struct, a scalar, or an
    /// untyped raw buffer.
    Object,
    /// A slice/array buffer with a known element count.
    Slice { len: Int<'ctx> },
    /// An external raw-pointer parameter. `size` is an unconstrained symbolic
    /// term (the caller may pass any allocation); nullability is *not* stored
    /// here — it is tracked via the pointer term (`term == 0`) path conditions.
    External,
}

/// The type of an allocation's contents.
///
/// This replaces the previous `Option<Ty>` (whose `None` conflated two distinct
/// cases): `Typed` carries a concrete `Ty` — the element type of a slice, the
/// object type of a `Box<T>`/struct, or `u8` for a raw byte buffer — while
/// `Generic` marks a symbolic element type whose concrete `Ty` cannot be
/// determined (e.g. `from_raw_parts::<T>`; its `size` uses the shared symbolic
/// `sizeof_T` and its length must be materialized via `set_slice_len`).
#[derive(Clone, Debug)]
pub(crate) enum ContentTy<'tcx> {
    Typed(Ty<'tcx>),
    Generic,
}

impl<'tcx> ContentTy<'tcx> {
    /// The concrete type, if known.
    pub(crate) fn as_ty(&self) -> Option<Ty<'tcx>> {
        match self {
            ContentTy::Typed(t) => Some(*t),
            ContentTy::Generic => None,
        }
    }

    /// Whether the element type is symbolic/unknown (no concrete `Ty`).
    pub(crate) fn is_generic(&self) -> bool {
        matches!(self, ContentTy::Generic)
    }
}

impl<'tcx> From<Option<Ty<'tcx>>> for ContentTy<'tcx> {
    fn from(o: Option<Ty<'tcx>>) -> Self {
        match o {
            Some(t) => ContentTy::Typed(t),
            None => ContentTy::Generic,
        }
    }
}

/// How an allocation's liveness is established, and for which region.
#[derive(Clone, Debug)]
pub(crate) enum Liveness<'tcx> {
    /// No liveness assumption. A VM-owned allocation is still alive while
    /// `!dead`; an external allocation with no assumption has unknown liveness
    /// (and fails `Alive` unless grounded in a live reference).
    Unassumed,
    /// Assumed alive for the named region (an `Alive(p, 'a)` contract/invariant,
    /// or `'static` for `ValidCStr` / `Allocated` params / `'static` data).
    AssumedFor(Region<'tcx>),
}

/// Uniform facts about the *pointer elements* of a container, established by
/// `x.iter()` for_each invariants.
///
/// Each fact is a property every element pointer satisfies.  The facts are
/// anchored to the container's allocation (not the container value) because a
/// pointer loaded from the container resolves its provenance through this
/// allocation, so the checker finds the fact via the loaded pointer's
/// provenance.
#[derive(Clone, Debug, Default)]
pub(crate) struct ForEachFacts<'ctx, 'tcx> {
    /// `Typed(iter(), T)`: every element pointer points at a valid `T`.
    pub target_ty: Option<Ty<'tcx>>,
    /// `Align(iter(), T)`: every element pointer is aligned to `align_of(T)`.
    pub aligned_ty: Option<Ty<'tcx>>,
    /// `Allocated(iter(), T, n)`: every element pointer backs `>= n` `T`
    /// elements (`n` may be symbolic).
    pub allocated: Option<(Ty<'tcx>, Int<'ctx>)>,
    /// `Owning(iter())`: every element pointer is the sole owner of its
    /// pointee (mutually non-aliasing).  Established by the trusted invariant;
    /// the aliasing *check* of the invariant itself is done by the alias
    /// analysis, not by this flag.
    pub owning: bool,
}

/// A memory allocation: a stack local, a heap object (`Box`/`Vec`), or an
/// external raw-pointer placeholder.
///
/// The allocation is stored in [`Memory::allocations`] at index `AllocId.0`
/// (an `AllocId` is a monotonic counter that doubles as the vector index).
#[derive(Clone, Debug)]
pub(crate) struct Allocation<'ctx, 'tcx> {
    /// Base address (fresh Z3 constant).
    pub base: Int<'ctx>,

    /// Size in bytes (Z3 term, may be symbolic).
    pub size: Int<'ctx>,

    /// Alignment in bytes (Z3 term, may be symbolic for a generic element
    /// type).
    pub align: Int<'ctx>,

    /// Type of the allocation's contents (concrete `Ty` or symbolic `Generic`).
    pub element_ty: ContentTy<'tcx>,

    /// The allocation shape (object vs slice vs external).
    pub kind: AllocKind<'ctx>,

    /// Whether the allocation has been freed (StorageDead / Drop).
    pub dead: bool,

    /// Whether the allocation's contents hold an initialized (readable) value:
    /// a heap constructor (`Box::new`, `Vec`), a reference parameter's referent,
    /// a callee return, a `write`, `ValidCStr`, or const/static byte data.
    /// Stays `false` for uninitialized memory (`Box::new_uninit`,
    /// `MaybeUninit::uninit`).
    pub initialized: bool,

    /// How this allocation's liveness is established, and for which region.
    /// Only consulted for external allocations, which carry no liveness
    /// guarantee (their memory is owned by the caller); `AssumedFor('a)` records
    /// the `Alive(p, 'a)` contract's region so the checker can reject a use that
    /// demands a longer region.
    pub liveness: Liveness<'tcx>,

    /// Whether the allocation is known to be a null-terminated byte buffer (a
    /// valid C string), asserted via a `ValidCStr` contract fact or struct
    /// invariant.
    pub nul_terminated: bool,

    /// Uniform facts about this allocation's pointer elements, established by
    /// `x.iter()` for_each invariants (`Typed`/`Align`/`Allocated`).
    pub for_each: ForEachFacts<'ctx, 'tcx>,

    /// The allocation a sub-view was derived from: a slice view created by
    /// `s[i..j]` / `s.get(range)`, `split_at` / `align_to` / `as_chunks`, or
    /// `from_raw_parts`. Each view is a fresh `AllocId` but a window onto the
    /// same memory as its source, so `root_alloc` follows this edge to group
    /// aliasing views. The address linkage back to the source lives in the
    /// pointer term (`parent_term + offset`) and provenance `offset`, not in
    /// base arithmetic.
    pub parent: Option<AllocId>,

    /// The *data* allocation of a two-part value, reached from the *header*
    /// allocation whose provenance the value carries. A `Vec`/`String`/`CString`
    /// value's provenance names its struct's stack slot, not the heap buffer it
    /// owns, so `as_ptr()`/`into_boxed_slice`/`drop`/mutation follow this edge
    /// to the buffer. A `&[T]` fat pointer's provenance already names the slice
    /// data, so this is redundant there.
    pub slice_data: Option<AllocId>,
}

impl<'ctx, 'tcx> Allocation<'ctx, 'tcx> {
    /// Construct a fresh allocation with all live/dead/invariant flags in
    /// their initial state.
    pub(crate) fn new(
        base: Int<'ctx>,
        size: Int<'ctx>,
        align: Int<'ctx>,
        element_ty: Option<Ty<'tcx>>,
        kind: AllocKind<'ctx>,
    ) -> Self {
        Allocation {
            base,
            size,
            align,
            element_ty: element_ty.into(),
            kind,
            dead: false,
            initialized: false,
            liveness: Liveness::Unassumed,
            nul_terminated: false,
            for_each: ForEachFacts::default(),
            parent: None,
            slice_data: None,
        }
    }

    /// Whether this allocation models an external raw-pointer parameter.
    pub(crate) fn is_external(&self) -> bool {
        matches!(self.kind, AllocKind::External)
    }

    /// The slice/array element count, if this allocation is slice data.
    ///
    /// Invariant: `size == len * sizeof(element_ty)` is maintained by the
    /// callers that call [`Self::set_slice_len`] (they compute `size` from the
    /// same `len` and push the equality as a path condition); it is *not*
    /// enforced here. `slice_len_from_value` falls back to `size / elem_size`
    /// when the length was never materialized.
    pub(crate) fn slice_len(&self) -> Option<&Int<'ctx>> {
        match &self.kind {
            AllocKind::Slice { len } => Some(len),
            _ => None,
        }
    }

    /// Mark this allocation as slice data with the given element count.
    /// Callers must keep `size == len * sizeof(element_ty)` consistent.
    pub(crate) fn set_slice_len(&mut self, len: Int<'ctx>) {
        self.kind = AllocKind::Slice { len };
    }
}

/// Per-path facts, read afterwards by the property checker.
///
/// Most flags are latched at most once during path execution (a contract fact
/// or a recognized discriminant / bounds check); `reenter` is instead derived
/// from the input path in [`VmState::new`].  They are per-path state, not
/// per-step: once set they are never cleared within a path.  (`has_checked_bounds`
/// is additionally accumulated *across checkpoints* by the engine, which reads
/// it back into the next path's flags.)
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PathFacts {
    /// Whether the current path re-enters a block (loop-unrolled), which lets
    /// the checker exempt the unrolled iteration's "second drop".
    pub reenter: bool,
    /// Whether a SplitTransmute contract was asserted by the caller.
    pub split_transmute_asserted: bool,
    /// Whether an `Alias` hazard was accepted via the caller's contract.
    pub alias_hazard_accepted: bool,
    /// Whether a ChecksIndexBoundsDisjoint call was processed in any
    /// checkpoint of this function (accumulated across checkpoints).
    pub has_checked_bounds: bool,
    /// Set once the path evaluated an `Iterator::next` discriminant whose
    /// variant was known symbolically.
    pub saw_next_discriminant: bool,
}

/// Per-byte symbolic state at a concrete offset in an allocation.
#[derive(Clone, Debug, Default)]
pub(crate) struct ByteInfo<'ctx> {
    /// Symbolic value, if tracked.
    pub value: Option<Int<'ctx>>,
    /// Whether the byte has been explicitly written.
    pub init: bool,
    /// NUL knowledge: `Some(true)` known NUL, `Some(false)` known non-NUL.
    pub nul: Option<bool>,
}

/// State for the two inlined-callee execution mechanisms, which are orthogonal:
///
/// * path replay (`CalleeEntry`/`CalleeExit` items) is iterative and keeps its
///   saved caller frames on `frames`;
/// * recursive inlining (`exec_inline_call`) unwinds via the Rust call stack,
///   is bounded by `inline_depth`, and stashes its per-call scratch bindings in
///   `arg_referents`/`deferred_field_writes`.
///
/// # Deferring `&mut` writes across the inline frame
///
/// While the callee runs, the caller's `local_fields` is parked in the saved
/// [`FrameSnapshot`], so a write through a `&mut` argument cannot land in it
/// immediately.  `arg_referents` pre-resolves (before `save_frame`) which caller
/// local each `&mut` argument points at, and `deferred_field_writes` collects
/// the writes to replay once the caller is restored:
///
/// ```text
/// struct Foo { field: i32 }
/// fn bar(foo: &mut Foo) { foo.field = 1; }   // inlined callee
/// fn main() {
///     let mut x = Foo { field: 0 };
///     bar(&mut x);                            // inline `bar`
/// }
/// ```
///
/// 1. Enter `bar`: `save_frame` parks `main`'s locals; `arg_referents[0] = x`.
/// 2. Run `bar`: `foo.field = 1` (`(*foo).field`) can no longer resolve `x` by
///    address (the caller's address map is gone), so it pushes `(x, [0], 1)`.
/// 3. Exit `bar`: `restore_frame` brings `main`'s locals back, then the deferred
///    write is replayed, giving `x.field == 1`.
#[derive(Default)]
pub(crate) struct InlineCtx<'ctx, 'tcx> {
    /// Current inlining depth (nested inlined callees), bounded by
    /// `MAX_INLINE_DEPTH`.  Distinct from `frames`: recursive inlining unwinds
    /// through the Rust call stack and does not push onto `frames`.
    pub inline_depth: usize,
    /// Stack of saved caller frames (a [`FrameSnapshot`] each) for inlined-callee
    /// path execution.
    pub frames: Vec<FrameSnapshot<'ctx, 'tcx>>,
    /// During `exec_inline_call`, maps each callee argument index to the
    /// *caller* local its value points at (resolved from the reference's
    /// address term before the caller's address map is saved away).  Used by
    /// `exec_assign` to resolve `(*self).field = val` writes through a `&mut
    /// self` reborrow temp back to the caller's referent.
    pub arg_referents: Vec<Option<Local>>,
    /// Field writes through a `&mut` argument collected during
    /// `exec_inline_call`, replayed against the caller's `local_fields` after
    /// `restore_frame` (the caller's field map is parked while the callee runs).
    /// Each entry is `(caller_local, field_path, value)`, where `caller_local`
    /// comes from `arg_referents` — the caller local the `&mut` argument points
    /// at, not the argument itself.
    pub deferred_field_writes: Vec<(Local, Vec<usize>, VmValue<'ctx, 'tcx>)>,
}

/// The operands a place was produced from, for guard inference (tracing a
/// switch/assert guard back to the pointer it null-checks or alignment-checks)
/// and division-axiom injection (following dataflow edges to reach `Div`/`Rem`
/// results).
///
/// The operator kind lets guard inference tell null-check guards (`Ne`) from
/// alignment/equality guards (`Eq`/`Rem`/`BitAnd`).
#[derive(Clone, Debug)]
pub(crate) struct BinaryOpSource {
    pub lhs: Option<PlaceKey>,
    pub rhs: Option<PlaceKey>,
    pub op: rustc_middle::mir::BinOp,
}

/// Per-frame analysis metadata keyed by `Local`/`PlaceKey`.
///
/// `op_sources` records operand sources for guard inference; `iter_ptr_offset`
/// is the cumulative `Iter`/`IterMut` field-0 pointer offset that keeps Z3
/// expressions compact.  Both are checkpointed together around a call via
/// [`VmState::save_frame`]/[`VmState::restore_frame`], so they share one struct
/// and one lifecycle.
#[derive(Default)]
pub(crate) struct AnalysisCtx {
    /// Operand sources for guard inference: destination → (lhs, rhs) place
    /// keys with the operator kind of the binary operation that produced the
    /// destination.
    pub(crate) op_sources: FxHashMap<PlaceKey, BinaryOpSource>,
}

/// The object space: every allocation plus the per-allocation contents that are
/// keyed purely by `AllocId` (fields and byte state).
///
/// This is the *address/place* layer — the memory that values live in — kept
/// separate from [`Locals`], which binds MIR locals (names) to values. An
/// `AllocId` doubles as the index into `allocations` (a fresh id is
/// `allocations.len()`), so the `AllocId`-keyed field/byte tables stay
/// consistent with the allocation vector.
#[derive(Default)]
pub(crate) struct Memory<'ctx, 'tcx> {
    /// All known allocations, indexed by `AllocId`.
    pub(crate) allocations: Vec<Allocation<'ctx, 'tcx>>,

    /// Per-byte symbolic state: (alloc_id, concrete_byte_offset) → ByteInfo.
    /// Populated by aggregate initialisation, pointer stores, and write call
    /// effects. Enables byte-level reasoning for properties like ValidCStr.
    pub(crate) bytes: FxHashMap<(AllocId, usize), ByteInfo<'ctx>>,

    /// Per-allocation field tracking: (alloc_id, viewed_type, field_indices) →
    /// value, i.e. the value of a field *within an allocation* viewed as
    /// `viewed_type`.  The `viewed_type` distinguishes reinterprets of the same
    /// allocation under different ADTs (e.g. `LeafNode` vs `InternalNode` cast
    /// views), so field index `1` resolves to `parent_idx` under `LeafNode` and
    /// `edges` under `InternalNode` without colliding.
    pub(crate) fields: FxHashMap<(AllocId, Ty<'tcx>, Vec<usize>), VmValue<'ctx, 'tcx>>,
}

/// Bindings from MIR locals (names) to values and places.
///
/// This is the *value* layer: `values` is the rvalue each local holds, `slots`
/// is the stack allocation backing each local's place (lvalue identity), and
/// `local_fields` is a local-keyed map of field values.  `local_fields` answers
/// "what value does the aggregate *bound to* `local` hold at field `path`?",
/// which is distinct from [`Memory::fields`], which answers "what value sits at
/// an address *within* an allocation viewed as a type?" (memory contents).
#[derive(Default)]
pub(crate) struct Locals<'ctx, 'tcx> {
    /// Current value bound to each MIR local (rvalue).
    pub(crate) values: FxHashMap<Local, VmValue<'ctx, 'tcx>>,

    /// The stack allocation backing each local's place (lvalue identity).
    pub(crate) slots: FxHashMap<Local, AllocId>,

    /// Field-level value tracking for aggregates: (local, field_indices) → value.
    /// Example: `(local_3, [0])` is `local_3.0`, `(local_3, [0, 1])` is `local_3.0.1`.
    /// This is the binding-value layer; see [`Memory::fields`] for the
    /// alloc-keyed memory-contents layer (pointee decomposition).
    pub(crate) local_fields: FxHashMap<(Local, Vec<usize>), VmValue<'ctx, 'tcx>>,

    /// Move-alias chain: `local → source` for a whole-place move (`_3 = move _4`).
    /// Lets `Owning` tell the call's own rebuilt owner (`boxed = move dest`) from
    /// a *previous* call's owner (also a shallow field, but tracing to a
    /// different destination).  Frame-scoped like the rest of [`Locals`], so it
    /// is saved/restored on inline entry/exit.
    pub(crate) move_sources: FxHashMap<Local, Local>,
}

/// Shared symbolic layout constants for generic types whose concrete layout is
/// unknown at verification time (an unconstrained `T`).
///
/// One constant per type keeps `ptr.add` strides, `access_bytes` element sizes,
/// and allocation sizes consistent so that SMT can cancel the `S` factor in
/// `InBound` (`(mid+n)·S <= len·S ⟺ mid+n <= len`); the alignment constant is
/// linked to the size by the layout constraint `sizeof_T % align_T == 0`.
#[derive(Default)]
pub(crate) struct LayoutCache<'ctx, 'tcx> {
    /// `sizeof_T` for each generic type, one symbolic constant per type.
    pub(crate) sizes: FxHashMap<Ty<'tcx>, Int<'ctx>>,

    /// `align_T` for each generic type, one symbolic constant per type.
    pub(crate) aligns: FxHashMap<Ty<'tcx>, Int<'ctx>>,
}

/// Accumulated solver state for the current path.
///
/// `constraints` is the assertion stream fed to Z3; `layout` and
/// `not_mask_terms` are term caches that keep generic-layout constants and
/// alignment-mask terms shared across the path (so Z3 expressions stay compact
/// and the `S` factor cancels in `InBound`).  All three are path-scoped and
/// monotonic: they accumulate as the VM steps and are never reset within a
/// path (or across inlined frames).
#[derive(Default)]
pub(crate) struct SolverState<'ctx, 'tcx> {
    /// Accumulated solver constraints along the current path: branch/guard
    /// constraints (`SwitchInt`/`Assert`), API preconditions, and symbolic
    /// layout facts (`sizeof_T`, `align_T`).  Asserted into the solver by
    /// [`VmState::assert_all`] and by the property checker's feasibility
    /// queries.
    pub(crate) constraints: Vec<Bool<'ctx>>,

    /// The shared symbolic layout constants (`sizeof_T`, `align_T`) for generic
    /// types whose concrete layout is unknown at verification time.
    pub(crate) layout: LayoutCache<'ctx, 'tcx>,

    /// Terms that are the result of a bitwise `Not` (two's-complement mask).
    /// Used to recognize `x & !(align-1)` alignment patterns in BitAnd so we
    /// can derive `align = -mask` and emit linear bounds for the result.
    pub(crate) not_mask_terms: FxHashSet<Int<'ctx>>,
}

/// The frame-scoped subset of [`VmState`] captured when entering an inlined
/// callee and restored on exit.
///
/// Everything here is keyed by MIR `Local` (the callee reuses the caller's
/// local indices), so it must be swapped out for the duration of the callee's
/// execution and swapped back afterwards.  The path-scoped state (`memory`,
/// `solver`, `path_facts`) is deliberately *not* captured: it accumulates
/// across the whole path, including inlined frames.
pub(crate) struct FrameSnapshot<'ctx, 'tcx> {
    pub(crate) caller_def_id: DefId,
    pub(crate) locals: Locals<'ctx, 'tcx>,
    pub(crate) analysis: AnalysisCtx,
}

/// The full symbolic execution state at a program point.
///
/// Accumulates locals, allocations, and solver constraints as the VM steps
/// through retained MIR items. The Z3 context is borrowed so a single context
/// can be reused across property checks.
pub(crate) struct VmState<'ctx, 'tcx> {
    // ── Shared handles (passed in at run start; not execution state, but
    //    needed to create terms and query types during checking)
    /// Shared Z3 context.
    pub(crate) ctx: &'ctx Context,

    /// Compiler type context.
    pub(crate) tcx: TyCtxt<'tcx>,

    // ── Current function frame (swapped on inline entry/exit)
    /// The function whose body we execute (the MIR is derived via [`Self::body`]).
    pub(crate) caller_def_id: DefId,

    // ── Memory + local bindings (the two-state layers)
    /// The object space: allocations, per-byte state, and per-allocation fields.
    pub(crate) memory: Memory<'ctx, 'tcx>,

    /// Bindings from MIR locals (names) to values, stack slots, and move-alias
    /// sources.
    pub(crate) locals: Locals<'ctx, 'tcx>,

    // ── Inlined-callee execution state
    /// Inlined-callee execution state (depth, frame stack, and per-call
    /// temporary bindings swapped on entry / restored on exit).
    pub(crate) inline: InlineCtx<'ctx, 'tcx>,

    // ── Per-frame analysis metadata
    /// Operand sources for guard inference and iterator pointer offsets,
    /// checkpointed together around a call.
    pub(crate) analysis: AnalysisCtx,

    /// Cumulative ptr offset for Iter/IterMut field [0] (ptr), keyed by the
    /// iterator's `self` local. Unlike `analysis`, this is *path-scoped*: an
    /// iterator's offset accumulates across loop iterations and inlined
    /// `&mut self` frames (whose `self` is always local 1), so it must survive
    /// [`save_frame`]/[`restore_frame`]. `post_inc_start` advances the ptr by
    /// `n` elements, so we increment this offset instead of nesting symbolic
    /// additions (which keeps Z3 expressions compact).
    pub(crate) iter_ptr_offset: FxHashMap<Local, Int<'ctx>>,

    // ── Accumulated solver state
    /// Solver constraints and term caches accumulated along the current path.
    pub(crate) solver: SolverState<'ctx, 'tcx>,

    // ── Per-path facts
    /// Per-path facts (latched while stepping, or derived in [`Self::new`]),
    /// read by the property checker.
    pub(crate) path_facts: PathFacts,
}

impl<'ctx, 'tcx> VmState<'ctx, 'tcx> {
    /// Create a fresh VM state for executing a path.
    pub(crate) fn new(
        ctx: &'ctx Context,
        tcx: TyCtxt<'tcx>,
        path: &Path,
        caller_def_id: DefId,
    ) -> Self {
        // Derive the one path fact the checker needs after `run` (whether the
        // path re-enters a block); the raw `Path` itself is not kept.
        let reenter = path.reenters();
        Self {
            ctx,
            tcx,
            caller_def_id,
            memory: Memory::default(),
            locals: Locals::default(),
            inline: InlineCtx::default(),
            analysis: AnalysisCtx::default(),
            iter_ptr_offset: FxHashMap::default(),
            solver: SolverState::default(),
            path_facts: PathFacts {
                reenter,
                ..PathFacts::default()
            },
        }
    }

    /// The MIR body of the current function, derived from `caller_def_id`.
    pub(crate) fn body(&self) -> &'tcx Body<'tcx> {
        self.tcx.optimized_mir(self.caller_def_id)
    }

    /// Capture the frame-scoped state before switching to an inlined callee.
    ///
    /// This is the single source of truth for *what* is frame-scoped: the
    /// local bindings (`values`, `slots`, `local_fields`, `move_sources`) and
    /// the per-frame analysis metadata (`op_sources`), plus the current
    /// function identity.  `iter_ptr_offset` is deliberately *not* captured —
    /// it is path-scoped so an iterator's cumulative offset survives inlined
    /// `&mut self` frames and loop iterations.  Both inline mechanisms
    /// (`handle_callee_entry` in path replay and `exec_inline_call`) call this,
    /// so they can no longer drift apart.
    pub(crate) fn save_frame(&mut self) -> FrameSnapshot<'ctx, 'tcx> {
        FrameSnapshot {
            caller_def_id: self.caller_def_id,
            locals: std::mem::take(&mut self.locals),
            analysis: std::mem::take(&mut self.analysis),
        }
    }

    /// Restore the frame-scoped state after an inlined callee returns.
    pub(crate) fn restore_frame(&mut self, snapshot: FrameSnapshot<'ctx, 'tcx>) {
        self.caller_def_id = snapshot.caller_def_id;
        self.locals = snapshot.locals;
        self.analysis = snapshot.analysis;
    }

    /// Look up the value bound to a MIR local.
    pub(crate) fn local_value(&self, local: Local) -> Option<&VmValue<'ctx, 'tcx>> {
        self.locals.values.get(&local)
    }

    /// Bind a value to a MIR local.
    pub(crate) fn set_local(&mut self, local: Local, value: VmValue<'ctx, 'tcx>) {
        self.locals.values.insert(local, value);
    }

    /// Get the symbolic address of a MIR local (its stack allocation's base).
    pub(crate) fn local_address(&mut self, local: Local) -> Int<'ctx> {
        self.ensure_local_allocation(local);
        let id = self.locals.slots[&local];
        self.memory.allocations[id.0].base.clone()
    }

    /// Allocate a fresh symbolic object and return its ID and base address.
    pub(crate) fn allocate(
        &mut self,
        size: Int<'ctx>,
        align: Int<'ctx>,
        element_ty: Option<Ty<'tcx>>,
    ) -> (AllocId, Int<'ctx>) {
        self.allocate_internal(size, align, element_ty, AllocKind::Object)
    }

    /// Allocate a fresh external allocation (for raw-pointer parameters).
    /// External allocations may be null and have unlimited size.
    pub(crate) fn allocate_external(
        &mut self,
        size: Int<'ctx>,
        align: Int<'ctx>,
        element_ty: Option<Ty<'tcx>>,
    ) -> (AllocId, Int<'ctx>) {
        self.allocate_internal(size, align, element_ty, AllocKind::External)
    }

    /// Allocate a slice/array data allocation with a known (possibly symbolic)
    /// element count. Computes `size = len * elem_size` and materializes `len`
    /// in one step, so `size` and the slice length can never diverge (the
    /// `size == len * elem_size` invariant is established here instead of being
    /// re-derived by every caller).
    pub(crate) fn allocate_slice(
        &mut self,
        len: Int<'ctx>,
        elem_size: Int<'ctx>,
        align: Int<'ctx>,
        element_ty: Option<Ty<'tcx>>,
    ) -> (AllocId, Int<'ctx>) {
        let size = Int::mul(self.ctx, &[&len, &elem_size]);
        let (id, base) = self.allocate(size, align, element_ty);
        self.alloc_mut(id).set_slice_len(len);
        (id, base)
    }

    fn allocate_internal(
        &mut self,
        size: Int<'ctx>,
        align: Int<'ctx>,
        element_ty: Option<Ty<'tcx>>,
        kind: AllocKind<'ctx>,
    ) -> (AllocId, Int<'ctx>) {
        let id = AllocId(self.memory.allocations.len());
        let base = {
            let name = format!(
                "{}_{}",
                if matches!(kind, AllocKind::External) {
                    "ext"
                } else {
                    "heap"
                },
                id.0
            );
            Int::new_const(self.ctx, name.as_str())
        };
        let alloc = Allocation::new(base.clone(), size, align, element_ty, kind);
        self.memory.allocations.push(alloc);
        (id, base)
    }

    /// Indexed access to an allocation by its `AllocId` (the id is the index).
    pub(crate) fn alloc(&self, id: AllocId) -> &Allocation<'ctx, 'tcx> {
        &self.memory.allocations[id.0]
    }

    /// Mutable indexed access to an allocation by its `AllocId`.
    pub(crate) fn alloc_mut(&mut self, id: AllocId) -> &mut Allocation<'ctx, 'tcx> {
        &mut self.memory.allocations[id.0]
    }

    /// The ultimate root allocation, following `parent` chains (sub-allocations
    /// created by `from_raw_parts` / `split_at` / `as_chunks`, whose `parent`
    /// points at the allocation they were split from).
    pub(crate) fn root_alloc(&self, id: AllocId) -> AllocId {
        let mut cur = id;
        let mut guard = 0;
        while let Some(parent) = self.alloc(cur).parent {
            cur = parent;
            guard += 1;
            // A parent chain longer than the total allocation count means a
            // cycle; stop rather than loop forever.
            if guard > self.memory.allocations.len() {
                break;
            }
        }
        cur
    }

    /// Create a fresh symbolic Z3 int constant (globally unique, even across
    /// calls with the same prefix — `Z3_mk_fresh_const` auto-suffixes the name).
    pub(crate) fn fresh_int(&self, prefix: &str) -> Int<'ctx> {
        Int::fresh_const(self.ctx, prefix)
    }

    /// Get the value of a specific field within an aggregate local.
    pub(crate) fn field_value(&self, local: Local, path: &[usize]) -> Option<&VmValue<'ctx, 'tcx>> {
        self.locals.local_fields.get(&(local, path.to_vec()))
    }

    /// The field carrying an owned value's heap pointer (`Box.0.0`/`Vec.0.0`,
    /// and for nested owners like `String` the deeper data field). Prefer the
    /// canonical `[0, 0]`; fall back to the first field with heap provenance.
    pub(crate) fn owner_ptr_field(&self, local: Local) -> Option<&VmValue<'ctx, 'tcx>> {
        self.field_value(local, &[0, 0])
            .filter(|v| v.provenance_alloc_id().is_some())
            .or_else(|| {
                self.locals
                    .local_fields
                    .iter()
                    .find(|((l, _), v)| *l == local && v.provenance_alloc_id().is_some())
                    .map(|(_, v)| v)
            })
    }

    /// Set the value of a specific field within an aggregate local.
    pub(crate) fn set_field_value(
        &mut self,
        local: Local,
        path: Vec<usize>,
        value: VmValue<'ctx, 'tcx>,
    ) {
        self.locals.local_fields.insert((local, path), value);
    }

    /// The value at a field offset *within an allocation* viewed as `view_ty`.
    ///
    /// This is the memory-contents layer ([`Memory::fields`]), the counterpart
    /// to [`Self::field_value`]'s binding-value layer ([`Locals::local_fields`]):
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
    ) -> Option<&VmValue<'ctx, 'tcx>> {
        self.memory.fields.get(&(alloc_id, view_ty, path.to_vec()))
    }

    /// Record a per-byte symbolic value at a concrete offset in an allocation.
    pub(crate) fn record_byte_value(&mut self, alloc_id: AllocId, offset: usize, term: Int<'ctx>) {
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
    pub(crate) fn get_byte_value(&self, alloc_id: AllocId, offset: usize) -> Option<&Int<'ctx>> {
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
    pub(crate) fn alloc_byte_values(&self, alloc_id: AllocId) -> Vec<(usize, &Int<'ctx>)> {
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
        let infos: Vec<(usize, ByteInfo<'ctx>)> = self
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

    /// Assert path conditions and invariant constraints into a solver.
    pub(crate) fn assert_all(&self, solver: &z3::Solver<'ctx>) {
        for cond in &self.solver.constraints {
            solver.assert(cond);
        }
        let zero = Int::from_u64(self.ctx, 0);
        for alloc in &self.memory.allocations {
            if !alloc.is_external() {
                solver.assert(&alloc.base._eq(&zero).not());
            }
            solver.assert(&alloc.size.ge(&zero));
            if alloc.align.simplify().as_u64() != Some(1) {
                solver.assert(&alloc.base.rem(&alloc.align)._eq(&zero));
            }
        }

        for (_local, value) in self.locals.values.iter() {
            self.assert_value_constraints(solver, value);
        }
        for value in self.locals.local_fields.values() {
            self.assert_value_constraints(solver, value);
        }
    }

    /// Assert a single symbolic value's known invariant constraints.
    fn assert_value_constraints(&self, solver: &z3::Solver<'ctx>, value: &VmValue<'ctx, 'tcx>) {
        let zero = Int::from_u64(self.ctx, 0);
        if value.invariants.non_null {
            solver.assert(&value.term._eq(&zero).not());
        }
        if let Some(ref prov) = value.provenance {
            let alloc = self.alloc(prov.alloc_id);
            let expected = Int::add(self.ctx, &[&alloc.base, &prov.offset]);
            solver.assert(&value.term._eq(&expected));
        }
        if matches!(
            value.ty.kind(),
            rustc_middle::ty::TyKind::Uint(_)
                | rustc_middle::ty::TyKind::Bool
                | rustc_middle::ty::TyKind::Char
        ) {
            solver.assert(&value.term.ge(&zero));
        }
        if matches!(value.ty.kind(), rustc_middle::ty::TyKind::Bool) {
            let one = Int::from_u64(self.ctx, 1);
            solver.assert(&value.term.le(&one));
        }
        if matches!(value.ty.kind(), rustc_middle::ty::TyKind::Char) {
            let max = Int::from_u64(self.ctx, 0x10FFFF);
            solver.assert(&value.term.le(&max));
        }
    }
}

impl std::fmt::Debug for VmState<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VmState")
            .field("locals_count", &self.locals.values.len())
            .field("allocations_count", &self.memory.allocations.len())
            .field("constraints", &self.solver.constraints.len())
            .finish()
    }
}

// ── Shared value extraction ──────────────────────────────────────

impl<'ctx, 'tcx> VmState<'ctx, 'tcx> {
    /// Extract a VmValue from a MIR operand.
    pub(crate) fn value_of_operand(&self, operand: &Operand<'tcx>) -> VmValue<'ctx, 'tcx> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self
                .value_of_place(place)
                .unwrap_or_else(|| self.unknown_value_for_place(place)),
            Operand::Constant(constant) => {
                let text = format!("{:?}", constant.const_);
                // `size_of::<T>()` / `align_of::<T>()` lower to the
                // `SizedTypeProperties::SIZE`/`::ALIGN` associated consts for a
                // generic `T`.  Bind them to the shared symbolic `sizeof_T` /
                // `align_T` (created by `size_sym`/`align_sym` during
                // `init_parameters`) so they agree with allocation sizes and
                // pointer strides instead of being unrelated fresh constants.
                if let rustc_middle::mir::Const::Unevaluated(uneval, _) = constant.const_ {
                    let def_name = self.tcx.def_path_str(uneval.def);
                    let is_size = def_name.ends_with("SizedTypeProperties::SIZE");
                    let is_align = def_name.ends_with("SizedTypeProperties::ALIGN");
                    if (is_size || is_align) && !uneval.args.is_empty() {
                        let ty = uneval.args.type_at(0);
                        let term = if is_size {
                            self.size_sym_read(ty)
                        } else {
                            self.align_sym_read(ty)
                        };
                        return VmValue::new(term, constant.const_.ty());
                    }
                }
                let int_val = crate::helpers::mir_utils::eval_const_scalar_int(
                    self.tcx,
                    &constant.const_,
                    &text,
                );
                let field_offset = int_val.is_none()
                    && crate::helpers::mir_utils::offset_of_container(self.tcx, &constant.const_)
                        .is_some();
                let term = if let Some(v) = int_val {
                    if v < 0 {
                        Int::from_i64(self.ctx, v as i64)
                    } else {
                        Int::from_u64(self.ctx, v as u64)
                    }
                } else {
                    // Create a deterministic name for const generics so
                    // multiple uses of the same parameter share one term.
                    let name = format!("const_{}", text.replace([':', '#', ' '], "_"));
                    Int::new_const(self.ctx, name.as_str())
                };
                let ty = constant.const_.ty();
                VmValue {
                    term,
                    ty,
                    provenance: None,
                    invariants: ValueInvariants::default(),
                    field_offset,
                    discriminant: None,
                    bool_cond: None,
                }
            }
            #[cfg(rapx_ge_95)]
            Operand::RuntimeChecks(_) => VmValue::new(
                self.fresh_int("runtime_checks"),
                self.body().local_decls[Local::from_usize(0)].ty,
            ),
        }
    }

    /// Look up the value stored at a MIR place.
    pub(crate) fn value_of_place(&self, place: &Place<'tcx>) -> Option<VmValue<'ctx, 'tcx>> {
        if place.projection.is_empty() {
            return self.locals.values.get(&place.local).cloned();
        }
        let place_ty = place.ty(self.body(), self.tcx).ty;

        // Collect field indices from projections
        let field_path: Vec<usize> = place
            .projection
            .iter()
            .filter_map(|proj| match proj.kind() {
                ProjectionElem::Field(field_idx, _) => Some(field_idx.as_usize()),
                _ => None,
            })
            .collect();

        // If we have a pure field path (only Field / Downcast projections),
        // look up in the per-field value map first.  For `Option`/`ControlFlow`,
        // the variant's data is stored under the same field index as the enum
        // field (the discriminant is tracked separately, not in local_fields),
        // so `(x as Some).0` resolves to `local_fields[x][0]`.
        let is_pure_field = place.projection.iter().all(|p| {
            matches!(
                p.kind(),
                ProjectionElem::Field(..) | ProjectionElem::Downcast(..)
            )
        });
        let has_downcast = place
            .projection
            .iter()
            .any(|p| matches!(p.kind(), ProjectionElem::Downcast(..)));
        if !field_path.is_empty() && is_pure_field {
            if let Some(val) = self.field_value(place.local, &field_path).cloned() {
                return Some(val);
            }
            if !has_downcast {
                // Fallback: when the base local has provenance, propagate it
                // to field accesses. This handles pointer-wrapper types (Box,
                // Unique, NonNull) where accessing inner pointer fields yields
                // the same provenance as the container.
                if let Some(base_val) = self.locals.values.get(&place.local) {
                    if let Some(ref prov) = base_val.provenance {
                        return Some(VmValue {
                            term: base_val.term.clone(),
                            ty: place_ty,
                            provenance: Some(prov.clone()),
                            invariants: base_val.invariants.clone(),
                            field_offset: false,
                            discriminant: None,
                            bool_cond: None,
                        });
                    }
                }
                return None;
            }
            // A Downcast without a materialized field falls through to the
            // Deref+Field / multi-element fallback below, which returns the
            // base local (preserving the pre-Downcast behavior instead of
            // forcing a fresh value).
        }

        // For Deref+Field chains (e.g. (*self).ptr), strip the leading Deref
        // projection(s) and look up local_fields with the remaining field path.
        if !field_path.is_empty()
            && field_path.len() < place.projection.len()
            && place
                .projection
                .iter()
                .any(|p| matches!(p.kind(), ProjectionElem::Deref))
        {
            // Only Deref and Field projections — all non-Field must be Deref
            // (a full-range `Subslice` (`arr[..]`) is transparent: it converts
            // `[T; N]` → `[T]` without changing which field is selected, so
            // `(*ptr).edges[..]` still resolves to the `edges` field).
            let non_field_deref = place.projection.iter().all(|p| {
                matches!(
                    p.kind(),
                    ProjectionElem::Field(..)
                        | ProjectionElem::Deref
                        | ProjectionElem::Subslice { .. }
                )
            });
            if non_field_deref {
                if let Some(val) = self.field_value(place.local, &field_path).cloned() {
                    return Some(val);
                }
                // Resolve a Deref+Field access through the pointee allocation's
                // per-allocation field tracking (e.g. `(*leaf).len` → the
                // `LeafNode.len` field value materialized by
                // `decompose_pointee_fields`).  The viewed type (pointee) is part
                // of the key so reinterpret casts (e.g. `LeafNode` → `InternalNode`)
                // resolve to the right field view.
                if let Some(base_val) = self.locals.values.get(&place.local) {
                    if let Some(alloc_id) = base_val.provenance_alloc_id() {
                        let view_ty = crate::helpers::mir_utils::pointee_ty(base_val.ty)
                            .unwrap_or(base_val.ty);
                        if let Some(val) = self.load_field(alloc_id, view_ty, &field_path).cloned() {
                            return Some(val);
                        }
                    }
                }
            }
        }

        // Handle Deref + Field projections: follow the dereference chain to
        // get the pointee base, then apply field offsets.
        // E.g. `(*self).ptr` → Deref then Field(0).
        let mut base = self.locals.values.get(&place.local)?.clone();
        for proj in place.projection.iter() {
            match proj.kind() {
                ProjectionElem::Deref => {
                    base.ty = place_ty;
                }
                ProjectionElem::Field(_field_idx, _) => {
                    // Try to get the field value from the VM's field tracking
                    if !field_path.is_empty() {
                        if let Some(val) = self.field_value(place.local, &field_path).cloned() {
                            return Some(val);
                        }
                    }
                    // Fallback: return the base with updated type info
                    base.ty = place_ty;
                }
                _ => {}
            }
        }

        // Fall back to type-level resolution with single-element projections
        if place.projection.len() == 1 {
            if let Some(proj) = place.projection.first() {
                if let ProjectionElem::Index(local) = proj {
                    if let Some(ref prov) = base.provenance {
                        let alloc_id = prov.alloc_id;
                        let byte_vals: Vec<_> = self.alloc_byte_values(alloc_id);
                        if !byte_vals.is_empty() {
                            let inner_ty = match base.ty.kind() {
                                rustc_middle::ty::TyKind::Array(inner, _) => *inner,
                                _ => return Some(base.clone()),
                            };
                            let elem_sz = self.size_of_ty(inner_ty) as usize;
                            let step = elem_sz.max(1);
                            if let Some(index_val) = self.locals.values.get(local) {
                                if let Some(concrete_idx) = index_val.term.as_u64() {
                                    let offset = concrete_idx as usize * step;
                                    let term = self
                                        .get_byte_value(alloc_id, offset)
                                        .cloned()
                                        .unwrap_or_else(|| self.fresh_int("arr_elem"));
                                    return Some(VmValue {
                                        term,
                                        ty: place_ty,
                                        provenance: None,
                                        invariants: ValueInvariants::default(),
                                        field_offset: false,
                                        discriminant: None,
                                        bool_cond: None,
                                    });
                                } else {
                                    let mut chain = self.fresh_int("arr_elem");
                                    for (offset, term) in byte_vals.iter().rev() {
                                        let vidx = offset / step;
                                        let idx_term = Int::from_u64(self.ctx, vidx as u64);
                                        let cond = index_val.term._eq(&idx_term);
                                        chain = Bool::ite(&cond, term, &chain);
                                    }
                                    return Some(VmValue {
                                        term: chain,
                                        ty: place_ty,
                                        provenance: None,
                                        invariants: ValueInvariants::default(),
                                        field_offset: false,
                                        discriminant: None,
                                        bool_cond: None,
                                    });
                                }
                            }
                        }
                    }
                    return Some(base.clone());
                }
                match proj.kind() {
                    ProjectionElem::Deref => {
                        // A `*dest` load of a reference created from a field
                        // (`let r = &mut self.v`) should yield the field's
                        // *value* (materialized by `propagate_field_values_to_ref`
                        // at the empty field path), not the field's address.
                        if let Some(v) = self.field_value(place.local, &[]).cloned() {
                            return Some(v);
                        }
                        let mut val = base.clone();
                        val.ty = place_ty;
                        return Some(val);
                    }
                    ProjectionElem::Field(_field_idx, _field_ty) => {
                        let val = base.clone();
                        return Some(val);
                    }
                    _ => {
                        // Downcast or other unsupported projection: still return
                        // the base with updated type so provenance propagates.
                        let mut val = base.clone();
                        val.ty = place_ty;
                        return Some(val);
                    }
                }
            }
        }

        // For multi-element projections with Deref+Field or Downcast, return
        // the base value since we already traced through Deref above.
        if place.projection.len() > 1
            && place.projection.iter().any(|p| {
                matches!(
                    p.kind(),
                    ProjectionElem::Deref | ProjectionElem::Downcast(..)
                )
            })
        {
            let mut val = base;
            val.ty = place_ty;
            return Some(val);
        }

        None
    }

    /// Create an unknown value for a place.
    ///
    /// The value carries no `non_null` (or any other) assumption: a raw pointer
    /// whose provenance was lost may still be null, so assuming non-null here
    /// would let `NonNull`/null-guard checks pass unsoundly.
    pub(crate) fn unknown_value_for_place(&self, place: &Place<'tcx>) -> VmValue<'ctx, 'tcx> {
        let ty = place.ty(self.body(), self.tcx).ty;
        VmValue::new(self.fresh_int("unknown"), ty)
    }
}
