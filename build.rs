use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=GENERALIST_MLX_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=GENERALIST_MLX_C_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=GENERALIST_MLX_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=GENERALIST_MLX_DEPLOYMENT_TARGET");
    println!("cargo:rerun-if-env-changed=GENERALIST_MLX_ALLOW_STUB");
    println!("cargo:rerun-if-changed=native/mlx/shim.cpp");
    println!("cargo:rerun-if-changed=native/mlx/shim.h");
    println!("cargo:rustc-check-cfg=cfg(generalist_mlx_native_linked)");

    if env::var_os("CARGO_FEATURE_EMBEDDED_MLX_NATIVE").is_none() {
        return;
    }

    let native_variables = [
        "GENERALIST_MLX_SOURCE_DIR",
        "GENERALIST_MLX_C_SOURCE_DIR",
        "GENERALIST_MLX_BUILD_DIR",
    ];
    let configured = native_variables
        .iter()
        .filter(|variable| env::var_os(variable).is_some())
        .count();
    if configured == 0 {
        let apple_silicon = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
            && env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64");
        let explicit_stub = env::var("GENERALIST_MLX_ALLOW_STUB").as_deref() == Ok("1");
        if apple_silicon && !explicit_stub {
            panic!(
                "embedded-mlx-native has no native MLX build; run `make embedded` from the Generalist repository root"
            );
        }
        println!(
            "cargo:warning=embedded-mlx-native is compiling the explicitly unavailable runtime stub"
        );
        return;
    }
    if configured != native_variables.len() {
        panic!(
            "embedded-mlx-native requires GENERALIST_MLX_SOURCE_DIR, GENERALIST_MLX_C_SOURCE_DIR, and GENERALIST_MLX_BUILD_DIR together"
        );
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        panic!("embedded-mlx-native requires an Apple Silicon macOS target");
    }

    let mlx_source = required_directory("GENERALIST_MLX_SOURCE_DIR");
    let mlx_c_source = required_directory("GENERALIST_MLX_C_SOURCE_DIR");
    let mlx_build = required_directory("GENERALIST_MLX_BUILD_DIR");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must define OUT_DIR"));
    let object = output.join("generalist-mlx-shim.o");
    let archive = output.join("libgeneralist_mlx_shim.a");

    let mut compile = Command::new("xcrun");
    compile
        .arg("clang++")
        .args([
            "-std=c++20",
            "-O3",
            "-DNDEBUG",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg("-isystem")
        .arg(&mlx_source)
        .arg("-isystem")
        .arg(&mlx_c_source)
        .args(["-c", "native/mlx/shim.cpp", "-o"])
        .arg(&object);
    if let Some(target) = env::var_os("GENERALIST_MLX_DEPLOYMENT_TARGET") {
        compile.arg(format!("-mmacosx-version-min={}", target.to_string_lossy()));
    }
    run(&mut compile, "compile the Generalist MLX shim");

    let mut archive_command = Command::new("xcrun");
    archive_command
        .args(["ar", "rcs"])
        .arg(&archive)
        .arg(&object);
    run(&mut archive_command, "archive the Generalist MLX shim");

    let mlx_archive = mlx_build.join("_deps/mlx-build/libmlx.a");
    let mlxc_archive = mlx_build.join("libmlxc.a");
    let jaccl_archive = mlx_build.join("jaccl/libjaccl.a");
    let metallib = mlx_build.join("_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib");
    println!("cargo:rerun-if-changed={}", mlx_archive.display());
    println!("cargo:rerun-if-changed={}", mlxc_archive.display());
    println!("cargo:rerun-if-changed={}", jaccl_archive.display());
    println!("cargo:rerun-if-changed={}", metallib.display());
    for required in [&mlx_archive, &mlxc_archive, &jaccl_archive, &metallib] {
        if !required.is_file() {
            panic!(
                "required native MLX artifact is missing: {}",
                required.display()
            );
        }
    }

    // Keep the Metal kernels inside the executable rather than depending on
    // a machine-specific build directory at runtime. The provider writes
    // these bytes to a digest-addressed temporary file before MLX starts.
    let embedded_metallib = output.join("generalist-mlx.metallib");
    fs::copy(&metallib, &embedded_metallib).unwrap_or_else(|error| {
        panic!(
            "failed to stage {} as an embedded Metal library: {error}",
            metallib.display()
        )
    });
    println!(
        "cargo:rustc-env=GENERALIST_EMBEDDED_MLX_METALLIB_PATH={}",
        embedded_metallib.display()
    );

    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-search=native={}", mlx_build.display());
    println!(
        "cargo:rustc-link-search=native={}",
        mlx_build.join("_deps/mlx-build").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        mlx_build.join("jaccl").display()
    );
    println!("cargo:rustc-link-lib=static=generalist_mlx_shim");
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=jaccl");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-cfg=generalist_mlx_native_linked");
}

fn required_directory(variable: &str) -> PathBuf {
    let value = env::var_os(variable)
        .unwrap_or_else(|| panic!("{variable} is required by embedded-mlx-native"));
    let directory = PathBuf::from(value);
    if !directory.is_dir() {
        panic!("{variable} is not a directory: {}", directory.display());
    }
    directory
}

fn run(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to {description}: {error}"));
    if !status.success() {
        panic!("failed to {description}: process exited with {status}");
    }
}
