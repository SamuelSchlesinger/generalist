use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::SeekFrom;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

const DEFAULT_MAX_BYTES: u64 = 32 * 1024;
const ABSOLUTE_MAX_BYTES: u64 = 1024 * 1024;

pub struct ReadFileTool;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileInput {
    path: String,
    #[serde(default)]
    offset_bytes: u64,
    #[serde(default = "default_max_bytes")]
    max_bytes: u64,
}

const fn default_max_bytes() -> u64 {
    DEFAULT_MAX_BYTES
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a bounded UTF-8 chunk from a regular file. Large files include a continuation byte offset."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The regular file path to read from"
                },
                "offset_bytes": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Byte offset at which to start (default: 0; use the continuation offset returned by a prior read)"
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 4,
                    "maximum": ABSOLUTE_MAX_BYTES,
                    "description": "Maximum bytes to return (default: 32768; max: 1048576)"
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let input: ReadFileInput = serde_json::from_value(input).map_err(|error| {
            Error::Other(format!(
                "Invalid read_file input: {error}. Example: {{\"path\":\"README.md\"}}"
            ))
        })?;
        if !(4..=ABSOLUTE_MAX_BYTES).contains(&input.max_bytes) {
            return Err(Error::Other(format!(
                "'max_bytes' must be between 4 and {ABSOLUTE_MAX_BYTES}"
            )));
        }

        read_utf8_chunk(&input.path, input.offset_bytes, input.max_bytes).await
    }
}

async fn read_utf8_chunk(path: &str, offset: u64, limit: u64) -> Result<String> {
    let mut options = tokio::fs::OpenOptions::new();
    options.read(true).custom_flags(libc::O_NONBLOCK);
    let mut file = options
        .open(path)
        .await
        .map_err(|error| Error::Other(format!("Failed to open file: {error}")))?;
    let metadata = file
        .metadata()
        .await
        .map_err(|error| Error::Other(format!("Failed to inspect file: {error}")))?;
    if !metadata.file_type().is_file() {
        return Err(Error::Other(
            "read_file only accepts regular files".to_string(),
        ));
    }
    let total_bytes = metadata.len();
    if offset > total_bytes {
        return Err(Error::Other(format!(
            "'offset_bytes' ({offset}) is past the end of the file ({total_bytes} bytes)"
        )));
    }

    file.seek(SeekFrom::Start(offset))
        .await
        .map_err(|error| Error::Other(format!("Failed to seek in file: {error}")))?;
    let mut bytes = Vec::with_capacity(limit as usize + 1);
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|error| Error::Other(format!("Failed to read file: {error}")))?;
    let read_past_limit = bytes.len() > limit as usize;
    bytes.truncate(limit as usize);

    let valid_bytes = match std::str::from_utf8(&bytes) {
        Ok(_) => bytes.len(),
        Err(error) if error.error_len().is_none() => error.valid_up_to(),
        Err(error) => {
            return Err(Error::Other(format!(
                "File is not valid UTF-8 at byte {}",
                offset + error.valid_up_to() as u64
            )))
        }
    };
    bytes.truncate(valid_bytes);
    let text = String::from_utf8(bytes)
        .expect("the byte prefix was checked as valid UTF-8 immediately above");
    let next_offset = offset
        .checked_add(valid_bytes as u64)
        .ok_or_else(|| Error::Other("File offset overflowed u64".to_string()))?;
    let has_more = read_past_limit || next_offset < total_bytes;

    if offset == 0 && !has_more {
        Ok(text)
    } else if has_more {
        Ok(format!(
            "[Read bytes {offset}..{next_offset} of {total_bytes}; continue with offset_bytes={next_offset}.]\n{text}"
        ))
    } else {
        Ok(format!(
            "[Read bytes {offset}..{next_offset} of {total_bytes}; end of file.]\n{text}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn small_file_is_returned_exactly() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"small file\n").unwrap();

        assert_eq!(
            ReadFileTool
                .execute(json!({"path": file.path()}))
                .await
                .unwrap(),
            "small file\n"
        );
    }

    #[tokio::test]
    async fn large_file_is_bounded_and_losslessly_paginated() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all("abcéxyz".as_bytes()).unwrap();

        let first = ReadFileTool
            .execute(json!({"path": file.path(), "max_bytes": 4}))
            .await
            .unwrap();
        assert_eq!(
            first,
            "[Read bytes 0..3 of 8; continue with offset_bytes=3.]\nabc"
        );

        let second = ReadFileTool
            .execute(json!({
                "path": file.path(),
                "offset_bytes": 3,
                "max_bytes": 5
            }))
            .await
            .unwrap();
        assert_eq!(second, "[Read bytes 3..8 of 8; end of file.]\néxyz");
    }

    #[tokio::test]
    async fn oversized_request_is_rejected_before_reading() {
        let error = ReadFileTool
            .execute(json!({
                "path": "/dev/zero",
                "max_bytes": ABSOLUTE_MAX_BYTES + 1
            }))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("must be between"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn non_regular_files_are_rejected() {
        let error = ReadFileTool
            .execute(json!({"path": "/dev/null"}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("regular files"));
    }
}
