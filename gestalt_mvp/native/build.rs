use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    
    // Generate C bindings for flutter_rust_bridge
    cbindgen::generate(&crate_dir)
        .expect("Unable to generate bindings")
        .write_to_file("src/ffi.h");
    
    // Rebuild if source files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/graph.rs");
    println!("cargo:rerun-if-changed=src/matcher.rs");
    println!("cargo:rerun-if-changed=src/fsm.rs");
    println!("cargo:rerun-if-changed=src/confidence.rs");
    println!("cargo:rerun-if-changed=src/guards.rs");
    println!("cargo:rerun-if-changed=src/message_bus.rs");
}
