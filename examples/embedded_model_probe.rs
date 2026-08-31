use generalist::provider::embedded::{ModelComponent, ModelPack};
use std::env;
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let manifest = required_path(arguments.next(), "MANIFEST")?;
    let blob_root = required_path(arguments.next(), "BLOB_ROOT")?;
    let tensor_name = arguments
        .next()
        .ok_or("missing TENSOR_NAME argument")?
        .into_string()
        .map_err(|_| "TENSOR_NAME must be valid UTF-8")?;
    if arguments.next().is_some() {
        return Err("usage: embedded_model_probe MANIFEST BLOB_ROOT TENSOR_NAME".into());
    }

    let pack = ModelPack::open_ollama(manifest, blob_root)?;
    let mut language = 0_usize;
    let mut mtp = 0_usize;
    let mut vision = 0_usize;
    let mut other = 0_usize;
    for tensor in pack.tensors() {
        match tensor.component {
            ModelComponent::Language => language += 1,
            ModelComponent::Mtp => mtp += 1,
            ModelComponent::Vision => vision += 1,
            ModelComponent::Other => other += 1,
        }
    }

    let config = pack.read_json_document("config.json")?;
    let architecture = config
        .get("architectures")
        .and_then(|architectures| architectures.get(0))
        .and_then(|architecture| architecture.as_str())
        .unwrap_or("unknown");
    let inspection = pack.inspect_tensor(&tensor_name)?;
    pack.verify_tensor(&tensor_name)?;

    println!("manifest={}", pack.manifest_source());
    println!("architecture={architecture}");
    println!(
        "tensor_blobs={} language={} mtp={} vision={} other={}",
        pack.tensors().len(),
        language,
        mtp,
        vision,
        other
    );
    println!(
        "selected={} payload_bytes={} arrays={} sha256=verified",
        inspection.declared_name,
        inspection.payload_bytes,
        inspection.tensors.len()
    );
    for tensor in inspection.tensors {
        println!(
            "array={} dtype={} shape={:?} offsets={:?}",
            tensor.name, tensor.dtype, tensor.shape, tensor.data_offsets
        );
    }
    Ok(())
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing {name} argument").into())
}
