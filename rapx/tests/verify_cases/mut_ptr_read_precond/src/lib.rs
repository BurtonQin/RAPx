#![feature(register_tool)]
#![register_tool(rapx)]
#![allow(dead_code)]
#![allow(unused_variables)]

// Regression: the builtin contract for `core::ptr::mut_ptr::read` referenced a
// nonexistent `src` parameter (copied from the free function `ptr::read`), so
// caller-side ValidPtr preconds were dropped and this trivially safe read
// reported UNSOUND. With the contract fixed to bind on `self`, the call must
// verify as SOUND.
#[rapx::verify]
#[rapx::requires(ValidPtr(ptr, u32, 1), kind = "precond")]
#[rapx::requires(Align(ptr, u32), kind = "precond")]
unsafe fn read_slot(ptr: *mut u32) -> u32 {
    unsafe { ptr.read() }
}
