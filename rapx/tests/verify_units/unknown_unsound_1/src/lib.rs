#![feature(register_tool)]
#![register_tool(rapx)]
#![allow(dead_code)]

// A callee with no safety contract: it is not a std function, carries no
// `#[rapx::requires]`, and is not itself a `#[rapx::verify]` target.  Its call
// site therefore cannot be discharged, and RAPx reports it as `Unknown` (rather
// than silently trusting it), making the caller UNSOUND.
unsafe fn unannotated_callee(ptr: *const u32) -> u32 {
    *ptr
}

// UNSOUND: the checkpoint calls `unannotated_callee`, which has no contract.
#[rapx::verify]
pub fn unsound_callsite_without_contract(ptr: *const u32) -> u32 {
    unsafe { unannotated_callee(ptr) }
}
