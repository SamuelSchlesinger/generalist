//! Conventions shared by every subprocess the host runs for the model
//! (`bash` commands and code-mode scripts), so their timeout bounds, output
//! handling, and result formats cannot drift apart.

use std::collections::VecDeque;
use std::process::ExitStatus;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 120;
pub(crate) const MAX_TIMEOUT_SECS: u64 = 600;

/// In-memory bytes retained per stream while a child runs. A `yes`-style
/// command must never grow the agent's memory without bound.
const TAIL_CAP_BYTES: usize = 512 * 1024;
/// On-disk spill cap per stream; beyond it the spill stops growing (the
/// in-memory tail keeps tracking the stream's end) so a runaway command
/// cannot exhaust the disk either.
const SPILL_CAP_BYTES: u64 = 64 * 1024 * 1024;

/// One fully drained child stream: a bounded tail plus an optional spill of
/// the full stream on disk.
pub(crate) struct CollectedStream {
    tail: VecDeque<u8>,
    total_bytes: u64,
    spill: Option<Spill>,
}

/// A spill remains temporary until its path is actually disclosed through
/// [`CollectedStream::into_text`]. Dropped/cancelled collectors therefore do
/// not strand files the caller never learned how to clean up.
struct Spill {
    path: tempfile::TempPath,
    beginning_only: bool,
}

impl CollectedStream {
    pub(crate) fn into_text(mut self) -> String {
        let tail_len = self.tail.len();
        let tail = self.tail.into_iter().collect::<Vec<_>>();
        let tail = String::from_utf8_lossy(&tail);
        if self.total_bytes <= tail_len as u64 {
            return tail.into_owned();
        }
        let saved = match self.spill.take() {
            Some(spill) => match spill.path.keep() {
                Ok(path) if spill.beginning_only => {
                    format!(" The stream's beginning was saved to: {}", path.display())
                }
                Ok(path) => format!(" Full stream saved to: {}", path.display()),
                Err(_) => String::new(),
            },
            None => String::new(),
        };
        format!(
            "[Stream too large: showing the last {} of {} bytes.{}]\n{}",
            tail_len, self.total_bytes, saved, tail
        )
    }
}

/// Drain a child stream to EOF while keeping memory and disk bounded.
///
/// The stream is always fully drained — a child must never block on a full
/// pipe — but only the last [`TAIL_CAP_BYTES`] stay in memory. When the
/// stream first outgrows the tail, everything seen so far spills to a temp
/// file that continues to grow up to [`SPILL_CAP_BYTES`]. The file is kept
/// only if [`CollectedStream::into_text`] reports its path. Read errors end
/// collection with whatever arrived, matching child-exit semantics.
pub(crate) async fn collect_bounded(
    mut reader: impl AsyncRead + Unpin,
    spill_prefix: &str,
) -> CollectedStream {
    let mut tail: VecDeque<u8> = VecDeque::with_capacity(TAIL_CAP_BYTES);
    let mut total: u64 = 0;
    let mut spill: Option<(tempfile::TempPath, tokio::fs::File)> = None;
    let mut spill_written: u64 = 0;
    let mut spill_truncated = false;
    let mut buf = vec![0u8; 16 * 1024];
    loop {
        let n = match reader.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        total += n as u64;
        if spill.is_none() && tail.len() + n > TAIL_CAP_BYTES {
            // First overflow: start the spill with everything seen so far.
            // A failed spill setup degrades to tail-only, never unbounded.
            let prefix = spill_prefix.to_string();
            let created = tokio::task::spawn_blocking(move || {
                let temporary = tempfile::Builder::new()
                    .prefix(&prefix)
                    .suffix(".txt")
                    .tempfile()?;
                Ok::<_, std::io::Error>(temporary.into_parts())
            })
            .await;
            if let Ok(Ok((file, path))) = created {
                let mut file = tokio::fs::File::from_std(file);
                let retained = tail.iter().copied().collect::<Vec<_>>();
                if file.write_all(&retained).await.is_ok() {
                    spill_written = tail.len() as u64;
                    spill = Some((path, file));
                }
            }
        }
        if let Some((_, file)) = spill.as_mut() {
            let allowed = (SPILL_CAP_BYTES.saturating_sub(spill_written) as usize).min(n);
            if allowed > 0 {
                if file.write_all(&buf[..allowed]).await.is_ok() {
                    spill_written += allowed as u64;
                } else {
                    // Stop retrying a failed file and describe it only as a
                    // beginning fragment if the path is later disclosed.
                    spill_written = SPILL_CAP_BYTES;
                    spill_truncated = true;
                }
            }
            if allowed < n {
                spill_truncated = true;
            }
        }
        tail.extend(buf[..n].iter().copied());
        if tail.len() > TAIL_CAP_BYTES {
            let excess = tail.len() - TAIL_CAP_BYTES;
            tail.drain(..excess);
        }
    }
    if let Some((_, file)) = spill.as_mut() {
        if file.flush().await.is_err() {
            spill_truncated = true;
        }
    }
    CollectedStream {
        tail,
        total_bytes: total,
        spill: spill.map(|(path, _)| Spill {
            path,
            beginning_only: spill_truncated,
        }),
    }
}

/// Kill the child's entire process group, then reap the leader normally.
///
/// Callers must have spawned the child with `process_group(0)`; without a
/// group kill, grandchildren of `bash -c 'x | y &'` survive a timeout.
pub(crate) fn kill_process_group(child: &tokio::process::Child) {
    if let Some(pid) = child.id() {
        // A negative pid addresses the whole process group.
        unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
    }
}

/// Combine a finished process's streams into the single text shown to the
/// model: quiet success stays bare, stderr is labeled, and a failure leads
/// with the exit code.
pub(crate) fn combine_output(status: ExitStatus, stdout: &str, stderr: &str) -> String {
    if status.success() && stderr.is_empty() {
        stdout.to_string()
    } else if status.success() {
        format!("{stdout}\nStderr:\n{stderr}")
    } else {
        format!(
            "Exit code: {}\nStdout:\n{}\nStderr:\n{}",
            status.code().unwrap_or(-1),
            stdout,
            stderr
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    #[tokio::test]
    async fn small_streams_pass_through_unchanged() {
        let data = b"hello world".as_slice();
        let collected = collect_bounded(data, "generalist-test-").await;
        assert_eq!(collected.into_text(), "hello world");
    }

    #[tokio::test]
    async fn oversized_streams_keep_a_bounded_tail_and_spill_fully() {
        let mut data = vec![b'a'; TAIL_CAP_BYTES * 2];
        data.extend_from_slice(b"THE_END");
        let total = data.len();
        let collected = collect_bounded(data.as_slice(), "generalist-test-").await;

        assert_eq!(collected.total_bytes, total as u64);
        assert!(collected.tail.len() <= TAIL_CAP_BYTES);
        let spill = collected.spill.as_ref().expect("stream should spill");
        assert!(!spill.beginning_only);
        let path = spill.path.to_path_buf();
        let spilled = std::fs::read(&path).unwrap();
        assert_eq!(spilled.len(), total, "spill must hold the full stream");
        let text = collected.into_text();
        assert!(text.starts_with("[Stream too large"));
        assert!(text.ends_with("THE_END"));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn undisclosed_spills_are_removed_when_collection_is_dropped() {
        let data = vec![b'x'; TAIL_CAP_BYTES + 1];
        let collected = collect_bounded(data.as_slice(), "generalist-test-").await;
        let path = collected
            .spill
            .as_ref()
            .expect("stream should spill")
            .path
            .to_path_buf();
        assert!(path.exists());

        drop(collected);
        assert!(!path.exists());
    }

    #[test]
    fn combine_output_formats() {
        let ok = ExitStatus::from_raw(0);
        assert_eq!(combine_output(ok, "out", ""), "out");
        assert_eq!(combine_output(ok, "out", "warn"), "out\nStderr:\nwarn");

        let failed = ExitStatus::from_raw(3 << 8);
        let combined = combine_output(failed, "out", "err");
        assert!(combined.starts_with("Exit code: 3\n"));
        assert!(combined.contains("Stdout:\nout"));
        assert!(combined.contains("Stderr:\nerr"));
    }
}
