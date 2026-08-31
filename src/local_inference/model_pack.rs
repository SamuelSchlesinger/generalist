//! Strict indexing of the content-addressed model pack used during embedded
//! Qwen development.
//!
//! The first adapter accepts Ollama's OCI-style manifest so Generalist can
//! differentially test the exact installed weights without copying 18 GB.
//! The rest of the engine consumes the normalized [`ModelPack`] API rather
//! than depending on Ollama's process, HTTP API, or manifest implementation.

use crate::error::{Error, Result};
use ring::digest::{Context, SHA256};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::str::FromStr;

const OCI_MANIFEST_MEDIA_TYPE: &str = "application/vnd.docker.distribution.manifest.v2+json";
const TENSOR_MEDIA_TYPE: &str = "application/vnd.ollama.image.tensor";
const JSON_MEDIA_TYPE: &str = "application/vnd.ollama.image.json";
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_JSON_DOCUMENT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SAFETENSOR_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const HASH_BUFFER_BYTES: usize = 1024 * 1024;

/// A parsed SHA-256 digest. Keeping bytes, rather than an unchecked string,
/// makes content-addressed paths incapable of containing path separators.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn hex(&self) -> String {
        let mut encoded = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            let _ = write!(encoded, "{byte:02x}");
        }
        encoded
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Sha256Digest")
            .field(&self.hex())
            .finish()
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "sha256:{}", self.hex())
    }
}

impl FromStr for Sha256Digest {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        let Some(hex) = value.strip_prefix("sha256:") else {
            return Err(pack_error("digest must start with `sha256:`"));
        };
        if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(pack_error(
                "SHA-256 digest must contain exactly 64 hexadecimal digits",
            ));
        }
        if hex.bytes().any(|byte| byte.is_ascii_uppercase()) {
            return Err(pack_error("SHA-256 digest must use lowercase hexadecimal"));
        }

        let mut digest = [0_u8; 32];
        for (index, pair) in hex.as_bytes().as_chunks::<2>().0.iter().enumerate() {
            digest[index] = (decode_hex(pair[0]) << 4) | decode_hex(pair[1]);
        }
        Ok(Self(digest))
    }
}

/// Logical part of the Qwen artifact. Text-only execution can exclude the
/// vision tower while the speculative decoder explicitly retains MTP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelComponent {
    Language,
    Mtp,
    Vision,
    Other,
}

/// One content-addressed file referenced by the model manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobDescriptor {
    pub media_type: String,
    pub digest: Sha256Digest,
    pub size: u64,
    pub name: Option<String>,
}

/// A named tensor layer in the normalized model index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub name: String,
    pub component: ModelComponent,
    pub blob: BlobDescriptor,
}

/// Metadata for an individual array inside a safetensors blob. Quantized
/// weights normally carry the packed weight, scale, and global scale together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

/// Validated safetensors header belonging to one manifest tensor layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorInspection {
    pub declared_name: String,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<TensorMetadata>,
    pub payload_bytes: u64,
}

/// Immutable normalized index over a content-addressed model artifact.
#[derive(Debug, Clone)]
pub struct ModelPack {
    manifest_path: Option<PathBuf>,
    manifest_source: String,
    blob_root: PathBuf,
    config: BlobDescriptor,
    tensors: BTreeMap<String, TensorDescriptor>,
    documents: BTreeMap<String, BlobDescriptor>,
    other_layers: Vec<BlobDescriptor>,
}

impl ModelPack {
    /// Open and structurally validate an Ollama OCI manifest and every blob it
    /// references. This performs bounded JSON parsing plus file type/size
    /// checks; cryptographic hashing remains explicit via [`Self::verify_blob`]
    /// so startup does not hash the complete 18 GB artifact on every run.
    pub fn open_ollama(
        manifest_path: impl AsRef<Path>,
        blob_root: impl AsRef<Path>,
    ) -> Result<Self> {
        let manifest_path = canonical_regular_file(manifest_path.as_ref(), "model manifest")?;
        let manifest_bytes =
            read_bounded_file(&manifest_path, MAX_MANIFEST_BYTES, "model manifest")?;
        let manifest_source = manifest_path.display().to_string();
        Self::from_ollama_manifest_bytes_inner(
            &manifest_bytes,
            manifest_source,
            Some(manifest_path),
            blob_root.as_ref(),
        )
    }

    /// Parse a manifest embedded with [`crate::include_model_pack!`] while keeping
    /// tensor payloads in an mmap-friendly content-addressed resource root.
    pub fn from_ollama_manifest_bytes(
        manifest: &[u8],
        source: impl Into<String>,
        blob_root: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::from_ollama_manifest_bytes_inner(manifest, source.into(), None, blob_root.as_ref())
    }

    fn from_ollama_manifest_bytes_inner(
        manifest_bytes: &[u8],
        manifest_source: String,
        manifest_path: Option<PathBuf>,
        blob_root: &Path,
    ) -> Result<Self> {
        if manifest_bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(pack_error(format!(
                "model manifest `{manifest_source}` exceeds the {MAX_MANIFEST_BYTES} byte limit"
            )));
        }
        let blob_root = fs::canonicalize(blob_root).map_err(|error| {
            pack_error(format!(
                "failed to resolve model blob directory {}: {error}",
                blob_root.display()
            ))
        })?;
        if !blob_root.is_dir() {
            return Err(pack_error(format!(
                "model blob root is not a directory: {}",
                blob_root.display()
            )));
        }

        let manifest: RawManifest = serde_json::from_slice(manifest_bytes)?;
        if manifest.schema_version != 2 {
            return Err(pack_error(format!(
                "unsupported model manifest schema version {}",
                manifest.schema_version
            )));
        }
        if manifest.media_type != OCI_MANIFEST_MEDIA_TYPE {
            return Err(pack_error(format!(
                "unsupported model manifest media type `{}`",
                manifest.media_type
            )));
        }
        if manifest.layers.is_empty() {
            return Err(pack_error("model manifest has no layers"));
        }

        let config = normalize_descriptor(manifest.config, false)?;
        let mut tensors = BTreeMap::new();
        let mut documents = BTreeMap::new();
        let mut other_layers = Vec::new();
        let mut named_layers = HashSet::new();

        for raw in manifest.layers {
            let descriptor = normalize_descriptor(raw, false)?;
            if let Some(name) = descriptor.name.as_deref() {
                validate_layer_name(name)?;
                if !named_layers.insert(name.to_owned()) {
                    return Err(pack_error(format!("duplicate named model layer `{name}`")));
                }
            }

            match descriptor.media_type.as_str() {
                TENSOR_MEDIA_TYPE => {
                    let name = descriptor
                        .name
                        .clone()
                        .ok_or_else(|| pack_error("tensor layer is missing its manifest name"))?;
                    tensors.insert(
                        name.clone(),
                        TensorDescriptor {
                            component: classify_tensor(&name),
                            name,
                            blob: descriptor,
                        },
                    );
                }
                JSON_MEDIA_TYPE => {
                    let name = descriptor
                        .name
                        .clone()
                        .ok_or_else(|| pack_error("JSON layer is missing its manifest name"))?;
                    documents.insert(name, descriptor);
                }
                _ => other_layers.push(descriptor),
            }
        }

        if tensors.is_empty() {
            return Err(pack_error("model manifest contains no tensors"));
        }

        let pack = Self {
            manifest_path,
            manifest_source,
            blob_root,
            config,
            tensors,
            documents,
            other_layers,
        };
        pack.validate_all_blob_metadata()?;
        Ok(pack)
    }

    pub fn manifest_path(&self) -> Option<&Path> {
        self.manifest_path.as_deref()
    }

    pub fn manifest_source(&self) -> &str {
        &self.manifest_source
    }

    pub fn blob_root(&self) -> &Path {
        &self.blob_root
    }

    pub fn config(&self) -> &BlobDescriptor {
        &self.config
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &TensorDescriptor> {
        self.tensors.values()
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorDescriptor> {
        self.tensors
            .get(name)
            .ok_or_else(|| pack_error(format!("model tensor `{name}` was not found")))
    }

    pub fn document(&self, name: &str) -> Result<&BlobDescriptor> {
        self.documents
            .get(name)
            .ok_or_else(|| pack_error(format!("model document `{name}` was not found")))
    }

    pub fn documents(&self) -> impl ExactSizeIterator<Item = (&str, &BlobDescriptor)> {
        self.documents
            .iter()
            .map(|(name, descriptor)| (name.as_str(), descriptor))
    }

    pub fn other_layers(&self) -> &[BlobDescriptor] {
        &self.other_layers
    }

    pub fn blob_path(&self, descriptor: &BlobDescriptor) -> PathBuf {
        self.blob_root
            .join(format!("sha256-{}", descriptor.digest.hex()))
    }

    /// Recheck that a blob is a regular non-symlink file of the declared size.
    pub fn validate_blob_metadata(&self, descriptor: &BlobDescriptor) -> Result<()> {
        let blob_path = self.blob_path(descriptor);
        let metadata = fs::symlink_metadata(&blob_path).map_err(|error| {
            pack_error(format!(
                "failed to inspect model blob {}: {error}",
                blob_path.display()
            ))
        })?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(pack_error(format!(
                "model blob must be a regular non-symlink file: {}",
                blob_path.display()
            )));
        }
        if metadata.len() != descriptor.size {
            return Err(pack_error(format!(
                "model blob {} has size {}, manifest declares {}",
                blob_path.display(),
                metadata.len(),
                descriptor.size
            )));
        }
        Ok(())
    }

    /// Hash one descriptor and compare it with its content-addressed name.
    /// Call this during installation/preflight or before first use of an
    /// untrusted pack, not once per token or once per warm process startup.
    pub fn verify_blob(&self, descriptor: &BlobDescriptor) -> Result<()> {
        self.validate_blob_metadata(descriptor)?;
        let blob_path = self.blob_path(descriptor);
        let mut file = File::open(&blob_path).map_err(|error| {
            pack_error(format!(
                "failed to open model blob {}: {error}",
                blob_path.display()
            ))
        })?;
        let mut context = Context::new(&SHA256);
        let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
        loop {
            let count = file.read(&mut buffer).map_err(|error| {
                pack_error(format!(
                    "failed to hash model blob {}: {error}",
                    blob_path.display()
                ))
            })?;
            if count == 0 {
                break;
            }
            context.update(&buffer[..count]);
        }
        let actual = context.finish();
        if actual.as_ref() != descriptor.digest.as_bytes() {
            return Err(pack_error(format!(
                "model blob {} does not match digest {}",
                blob_path.display(),
                descriptor.digest
            )));
        }
        Ok(())
    }

    pub fn verify_tensor(&self, name: &str) -> Result<()> {
        self.verify_blob(&self.tensor(name)?.blob)
    }

    /// Parse and validate the bounded safetensors header for a declared tensor
    /// blob without materializing its payload in Rust.
    pub fn inspect_tensor(&self, name: &str) -> Result<SafetensorInspection> {
        let tensor = self.tensor(name)?;
        self.validate_blob_metadata(&tensor.blob)?;
        let blob_path = self.blob_path(&tensor.blob);
        let mut file = File::open(&blob_path).map_err(|error| {
            pack_error(format!(
                "failed to open tensor blob {}: {error}",
                blob_path.display()
            ))
        })?;

        let mut length_bytes = [0_u8; 8];
        file.read_exact(&mut length_bytes).map_err(|error| {
            pack_error(format!(
                "failed to read safetensors header length from {}: {error}",
                blob_path.display()
            ))
        })?;
        let header_bytes = u64::from_le_bytes(length_bytes);
        if header_bytes == 0 || header_bytes > MAX_SAFETENSOR_HEADER_BYTES {
            return Err(pack_error(format!(
                "safetensors header in {} has invalid length {header_bytes}",
                blob_path.display()
            )));
        }
        let payload_offset = 8_u64
            .checked_add(header_bytes)
            .ok_or_else(|| pack_error("safetensors header length overflow"))?;
        if payload_offset > tensor.blob.size {
            return Err(pack_error(format!(
                "safetensors header in {} exceeds the blob size",
                blob_path.display()
            )));
        }

        let mut header = vec![0_u8; header_bytes as usize];
        file.read_exact(&mut header).map_err(|error| {
            pack_error(format!(
                "failed to read safetensors header from {}: {error}",
                blob_path.display()
            ))
        })?;
        let parsed: BTreeMap<String, Value> = serde_json::from_slice(&header)?;
        let payload_bytes = tensor.blob.size - payload_offset;
        let mut metadata = BTreeMap::new();
        let mut tensors = Vec::new();
        let mut declared_tensor_found = false;

        for (entry_name, entry) in parsed {
            if entry_name == "__metadata__" {
                let object = entry.as_object().ok_or_else(|| {
                    pack_error("safetensors __metadata__ entry must be an object")
                })?;
                for (key, value) in object {
                    let value = value
                        .as_str()
                        .ok_or_else(|| pack_error("safetensors metadata values must be strings"))?;
                    metadata.insert(key.clone(), value.to_owned());
                }
                continue;
            }

            if entry_name == tensor.name {
                declared_tensor_found = true;
            } else if !entry_name.starts_with(&format!("{}.", tensor.name)) {
                return Err(pack_error(format!(
                    "tensor blob declared as `{}` contains unrelated tensor `{entry_name}`",
                    tensor.name
                )));
            }
            tensors.push(parse_tensor_metadata(entry_name, entry, payload_bytes)?);
        }

        if !declared_tensor_found {
            return Err(pack_error(format!(
                "tensor blob does not contain its declared tensor `{}`",
                tensor.name
            )));
        }
        if tensors.is_empty() {
            return Err(pack_error("safetensors blob contains no arrays"));
        }
        validate_non_overlapping_offsets(&tensors)?;

        // Ensure the file was not truncated between metadata validation and
        // header parsing. Seeking is intentionally cheap and payload-neutral.
        let observed_size = file.seek(SeekFrom::End(0)).map_err(|error| {
            pack_error(format!(
                "failed to seek tensor blob {}: {error}",
                blob_path.display()
            ))
        })?;
        if observed_size != tensor.blob.size {
            return Err(pack_error(
                "tensor blob size changed while it was inspected",
            ));
        }

        Ok(SafetensorInspection {
            declared_name: tensor.name.clone(),
            metadata,
            tensors,
            payload_bytes,
        })
    }

    pub fn read_json_document(&self, name: &str) -> Result<Value> {
        let descriptor = self.document(name)?;
        self.validate_blob_metadata(descriptor)?;
        if descriptor.size > MAX_JSON_DOCUMENT_BYTES {
            return Err(pack_error(format!(
                "model JSON document `{name}` exceeds the {} byte limit",
                MAX_JSON_DOCUMENT_BYTES
            )));
        }
        let bytes = read_bounded_file(
            &self.blob_path(descriptor),
            MAX_JSON_DOCUMENT_BYTES,
            "model JSON document",
        )?;
        serde_json::from_slice(&bytes).map_err(Into::into)
    }

    fn validate_all_blob_metadata(&self) -> Result<()> {
        self.validate_blob_metadata(&self.config)?;
        for tensor in self.tensors.values() {
            self.validate_blob_metadata(&tensor.blob)?;
        }
        for descriptor in self.documents.values().chain(self.other_layers.iter()) {
            self.validate_blob_metadata(descriptor)?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct RawManifest {
    #[serde(rename = "schemaVersion")]
    schema_version: u32,
    #[serde(rename = "mediaType")]
    media_type: String,
    config: RawDescriptor,
    layers: Vec<RawDescriptor>,
}

#[derive(Debug, Deserialize)]
struct RawDescriptor {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
    size: u64,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensorMetadata {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn normalize_descriptor(raw: RawDescriptor, allow_empty: bool) -> Result<BlobDescriptor> {
    if raw.media_type.is_empty() {
        return Err(pack_error("model blob media type cannot be empty"));
    }
    if !allow_empty && raw.size == 0 {
        return Err(pack_error("model blob size cannot be zero"));
    }
    Ok(BlobDescriptor {
        media_type: raw.media_type,
        digest: raw.digest.parse()?,
        size: raw.size,
        name: raw.name,
    })
}

fn validate_layer_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(pack_error("named model layer cannot have an empty name"));
    }
    if name.contains('\0') || name.contains('/') || name.contains('\\') {
        return Err(pack_error(format!("invalid model layer name `{name}`")));
    }
    Ok(())
}

fn classify_tensor(name: &str) -> ModelComponent {
    let lowered = name.to_ascii_lowercase();
    if lowered.starts_with("mtp.") || lowered.contains(".mtp.") {
        ModelComponent::Mtp
    } else if lowered.contains("vision") || lowered.contains("visual") {
        ModelComponent::Vision
    } else if lowered == "lm_head.weight" || lowered.starts_with("model.language_model.") {
        ModelComponent::Language
    } else {
        ModelComponent::Other
    }
}

fn parse_tensor_metadata(name: String, value: Value, payload_bytes: u64) -> Result<TensorMetadata> {
    let raw: RawTensorMetadata = serde_json::from_value(value)?;
    if raw.dtype.is_empty() {
        return Err(pack_error(format!("tensor `{name}` has an empty dtype")));
    }
    if raw.data_offsets[0] > raw.data_offsets[1] || raw.data_offsets[1] > payload_bytes {
        return Err(pack_error(format!(
            "tensor `{name}` has invalid data offsets {:?} for a {payload_bytes} byte payload",
            raw.data_offsets
        )));
    }
    let expected_bytes = tensor_storage_bytes(&raw.dtype, &raw.shape)?;
    let observed_bytes = raw.data_offsets[1] - raw.data_offsets[0];
    if expected_bytes != observed_bytes {
        return Err(pack_error(format!(
            "tensor `{name}` shape/dtype requires {expected_bytes} bytes but offsets span {observed_bytes}"
        )));
    }
    Ok(TensorMetadata {
        name,
        dtype: raw.dtype,
        shape: raw.shape,
        data_offsets: raw.data_offsets,
    })
}

fn tensor_storage_bytes(dtype: &str, shape: &[u64]) -> Result<u64> {
    let bits_per_element = match dtype {
        "BOOL" | "I8" | "U8" | "F8_E4M3" | "F8_E5M2" => 8,
        "I16" | "U16" | "F16" | "BF16" => 16,
        "I32" | "U32" | "F32" => 32,
        "I64" | "U64" | "F64" => 64,
        other => {
            return Err(pack_error(format!(
                "unsupported safetensors dtype `{other}`"
            )))
        }
    };
    let elements = shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| pack_error("tensor shape element count overflow"))
    })?;
    elements
        .checked_mul(bits_per_element)
        .and_then(|bits| bits.checked_div(8))
        .ok_or_else(|| pack_error("tensor storage size overflow"))
}

fn validate_non_overlapping_offsets(tensors: &[TensorMetadata]) -> Result<()> {
    let mut regions: Vec<_> = tensors
        .iter()
        .map(|tensor| (tensor.data_offsets, tensor.name.as_str()))
        .collect();
    regions.sort_unstable_by_key(|(offsets, _)| offsets[0]);
    let mut previous_end = 0_u64;
    for (offsets, name) in regions {
        if offsets[0] < previous_end {
            return Err(pack_error(format!(
                "tensor `{name}` overlaps another array in its safetensors blob"
            )));
        }
        previous_end = offsets[1];
    }
    Ok(())
}

fn canonical_regular_file(path: &Path, description: &str) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        pack_error(format!(
            "failed to inspect {description} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(pack_error(format!(
            "{description} must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    fs::canonicalize(path).map_err(|error| {
        pack_error(format!(
            "failed to resolve {description} {}: {error}",
            path.display()
        ))
    })
}

fn read_bounded_file(path: &Path, limit: u64, description: &str) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path).map_err(|error| {
        pack_error(format!(
            "failed to inspect {description} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.len() > limit {
        return Err(pack_error(format!(
            "{description} {} exceeds the {limit} byte limit",
            path.display()
        )));
    }
    let mut file = File::open(path).map_err(|error| {
        pack_error(format!(
            "failed to open {description} {}: {error}",
            path.display()
        ))
    })?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes).map_err(|error| {
        pack_error(format!(
            "failed to read {description} {}: {error}",
            path.display()
        ))
    })?;
    Ok(bytes)
}

fn decode_hex(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("hex syntax was validated before decoding"),
    }
}

fn pack_error(message: impl Into<String>) -> Error {
    Error::Other(format!("embedded model pack: {}", message.into()))
}

/// Embed the small immutable model manifest at compile time while resolving
/// tensor payloads from a packaged content-addressed resource directory.
///
/// This intentionally avoids placing the 18 GB tensor payload in a Rust object
/// file. It preserves zero-copy/mmap behavior and lets release engineering
/// choose an app-bundle resource, external pack, or later custom Mach-O layout.
#[macro_export]
macro_rules! include_model_pack {
    ($manifest:expr, $blob_root:expr) => {{
        $crate::provider::embedded::ModelPack::from_ollama_manifest_bytes(
            include_bytes!($manifest),
            concat!(file!(), " -> ", stringify!($manifest)),
            $blob_root,
        )
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::digest;
    use serde_json::json;
    use std::io::Write;

    struct Fixture {
        _temp: tempfile::TempDir,
        manifest: PathBuf,
        blobs: PathBuf,
        tensor_name: String,
        tensor_digest: String,
    }

    fn digest_string(bytes: &[u8]) -> String {
        let value = digest::digest(&SHA256, bytes);
        let mut output = String::from("sha256:");
        for byte in value.as_ref() {
            use std::fmt::Write as _;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }

    fn write_blob(root: &Path, bytes: &[u8]) -> String {
        let digest = digest_string(bytes);
        let path = root.join(format!("sha256-{}", &digest[7..]));
        fs::write(path, bytes).unwrap();
        digest
    }

    fn safetensor(header: Value, payload: &[u8]) -> Vec<u8> {
        let mut encoded = serde_json::to_vec(&header).unwrap();
        while !encoded.len().is_multiple_of(8) {
            encoded.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + encoded.len() + payload.len());
        file.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        file.extend_from_slice(&encoded);
        file.extend_from_slice(payload);
        file
    }

    fn fixture() -> Fixture {
        let temp = tempfile::tempdir().unwrap();
        let blobs = temp.path().join("blobs");
        fs::create_dir(&blobs).unwrap();

        let config = b"{}";
        let config_digest = write_blob(&blobs, config);
        let tensor_name = "lm_head.weight".to_string();
        let tensor = safetensor(
            json!({
                "__metadata__": {"quant_type": "nvfp4"},
                "lm_head.weight": {
                    "dtype": "U32", "shape": [2, 1], "data_offsets": [0, 8]
                },
                "lm_head.weight.scale": {
                    "dtype": "U8", "shape": [2, 2], "data_offsets": [8, 12]
                },
                "lm_head.weight.global_scale": {
                    "dtype": "F32", "shape": [], "data_offsets": [12, 16]
                }
            }),
            &[0_u8; 16],
        );
        let tensor_digest = write_blob(&blobs, &tensor);
        let document = br#"{"architectures":["Qwen3_5ForConditionalGeneration"]}"#;
        let document_digest = write_blob(&blobs, document);
        let license = b"Apache License";
        let license_digest = write_blob(&blobs, license);

        let manifest_value = json!({
            "schemaVersion": 2,
            "mediaType": OCI_MANIFEST_MEDIA_TYPE,
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": config_digest,
                "size": config.len()
            },
            "layers": [
                {
                    "mediaType": TENSOR_MEDIA_TYPE,
                    "digest": tensor_digest,
                    "size": tensor.len(),
                    "name": tensor_name
                },
                {
                    "mediaType": JSON_MEDIA_TYPE,
                    "digest": document_digest,
                    "size": document.len(),
                    "name": "config.json"
                },
                {
                    "mediaType": "application/vnd.ollama.image.license",
                    "digest": license_digest,
                    "size": license.len()
                }
            ]
        });
        let manifest = temp.path().join("manifest.json");
        fs::write(&manifest, serde_json::to_vec(&manifest_value).unwrap()).unwrap();

        Fixture {
            _temp: temp,
            manifest,
            blobs,
            tensor_name,
            tensor_digest,
        }
    }

    #[test]
    fn opens_and_inspects_quantized_tensor_blob() {
        let fixture = fixture();
        let pack = ModelPack::open_ollama(&fixture.manifest, &fixture.blobs).unwrap();

        assert_eq!(pack.tensors().len(), 1);
        assert_eq!(
            pack.tensor(&fixture.tensor_name).unwrap().component,
            ModelComponent::Language
        );
        assert_eq!(pack.documents().len(), 1);
        assert_eq!(
            pack.read_json_document("config.json").unwrap()["architectures"][0],
            "Qwen3_5ForConditionalGeneration"
        );
        pack.verify_tensor(&fixture.tensor_name).unwrap();

        let inspection = pack.inspect_tensor(&fixture.tensor_name).unwrap();
        assert_eq!(inspection.metadata["quant_type"], "nvfp4");
        assert_eq!(inspection.tensors.len(), 3);
        assert!(inspection
            .tensors
            .iter()
            .any(|tensor| tensor.name == fixture.tensor_name && tensor.dtype == "U32"));
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        let fixture = fixture();
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&fixture.manifest).unwrap()).unwrap();
        let duplicate = manifest["layers"][0].clone();
        manifest["layers"].as_array_mut().unwrap().push(duplicate);
        fs::write(&fixture.manifest, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = ModelPack::open_ollama(&fixture.manifest, &fixture.blobs).unwrap_err();
        assert!(error.to_string().contains("duplicate named model layer"));
    }

    #[test]
    fn rejects_bad_digest_syntax_before_building_a_path() {
        let fixture = fixture();
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&fixture.manifest).unwrap()).unwrap();
        manifest["layers"][0]["digest"] = json!("sha256:../../escape");
        fs::write(&fixture.manifest, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = ModelPack::open_ollama(&fixture.manifest, &fixture.blobs).unwrap_err();
        assert!(error.to_string().contains("64 hexadecimal digits"));
    }

    #[test]
    fn rejects_blob_size_mismatch() {
        let fixture = fixture();
        let blob = fixture
            .blobs
            .join(format!("sha256-{}", &fixture.tensor_digest[7..]));
        let mut file = fs::OpenOptions::new().append(true).open(blob).unwrap();
        file.write_all(&[0]).unwrap();

        let error = ModelPack::open_ollama(&fixture.manifest, &fixture.blobs).unwrap_err();
        assert!(error.to_string().contains("manifest declares"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_blob() {
        use std::os::unix::fs::symlink;

        let fixture = fixture();
        let blob = fixture
            .blobs
            .join(format!("sha256-{}", &fixture.tensor_digest[7..]));
        let target = fixture.blobs.join("target");
        fs::rename(&blob, &target).unwrap();
        symlink(&target, &blob).unwrap();

        let error = ModelPack::open_ollama(&fixture.manifest, &fixture.blobs).unwrap_err();
        assert!(error.to_string().contains("regular non-symlink file"));
    }

    #[test]
    fn rejects_unrelated_array_in_tensor_blob() {
        let fixture = fixture();
        let tensor = safetensor(
            json!({
                "lm_head.weight": {
                    "dtype": "U8", "shape": [1], "data_offsets": [0, 1]
                },
                "other.weight": {
                    "dtype": "U8", "shape": [1], "data_offsets": [1, 2]
                }
            }),
            &[0_u8; 2],
        );
        let replacement_digest = write_blob(&fixture.blobs, &tensor);
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&fixture.manifest).unwrap()).unwrap();
        manifest["layers"][0]["digest"] = json!(replacement_digest);
        manifest["layers"][0]["size"] = json!(tensor.len());
        fs::write(&fixture.manifest, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let pack = ModelPack::open_ollama(&fixture.manifest, &fixture.blobs).unwrap();
        let error = pack.inspect_tensor(&fixture.tensor_name).unwrap_err();
        assert!(error.to_string().contains("contains unrelated tensor"));
    }

    #[test]
    fn compile_time_manifest_macro_opens_runtime_resource_pack() {
        let temp = tempfile::tempdir().unwrap();
        let blobs = temp.path().join("blobs");
        fs::create_dir(&blobs).unwrap();
        fs::write(
            blobs.join("sha256-44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"),
            b"{}",
        )
        .unwrap();
        fs::write(
            blobs.join("sha256-cf4051ef5db4ce6e38aa9e5e2bc54993328ac077a19a8db4a8beff792afed5a9"),
            b"tensor",
        )
        .unwrap();

        let pack = crate::include_model_pack!("testdata/embedded_manifest.json", &blobs).unwrap();
        assert_eq!(pack.tensors().len(), 1);
        assert!(pack.manifest_path().is_none());
        assert!(pack
            .manifest_source()
            .contains("testdata/embedded_manifest.json"));
    }
}
