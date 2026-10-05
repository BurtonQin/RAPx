fn main() {
    println!("cargo::rustc-check-cfg=cfg(rapx_rustc_ge_199)");
    println!("cargo::rustc-check-cfg=cfg(rapx_rustc_ge_196)");

    let version = rustc_version::version().unwrap();
    let minor = version.minor;

    if minor >= 99 {
        println!("cargo:rustc-cfg=rapx_rustc_ge_199");
    }
    // Historical name: the gated std form is already required on the pinned
    // 1.95 nightly, not only on 1.96+ releases.
    if minor >= 95 {
        println!("cargo:rustc-cfg=rapx_rustc_ge_196");
    }
}
