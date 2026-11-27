//! Writing a file whole or not at all, by a new file renamed over the old one.
//!
//! An in-place write (`std::fs::write`) truncates the file and writes into it. Two things go wrong with that here:
//! a crash or a full disk midway leaves half a file, and a file that is hard-linked into other folders (game folders
//! and their test copies share files that way) changes in every one of them at once. Writing a sibling and renaming it
//! over the path replaces only this directory entry: the other links keep what they had, and a reader sees the old
//! file or the new one, never a mix.

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Write `bytes` to `path` through a temporary sibling, synced, then renamed over `path`. Makes the parent folder if
/// it is missing. On any failure the temporary file is removed and `path` is left as it was.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
    }
    let name = path
        .file_name()
        .ok_or_else(|| format!("{} names no file", path.display()))?
        .to_string_lossy()
        .into_owned();
    let temporary = path.with_file_name(format!(
        ".{name}.{}-{}.tmp",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        file.write_all(bytes).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        drop(file);
        std::fs::rename(&temporary, path).map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::write_atomic;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("somnium-atomic-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The reason it exists: a file hard-linked into another folder is replaced here only, and the other link keeps
    /// what it had (an in-place write changed both).
    #[test]
    fn a_hard_linked_file_is_replaced_here_only() {
        let dir = scratch("link");
        let here = dir.join("scene.json");
        let there = dir.join("copy_of_scene.json");
        std::fs::write(&here, b"old").unwrap();
        std::fs::hard_link(&here, &there).unwrap();
        write_atomic(&here, b"new").unwrap();
        assert_eq!(std::fs::read(&here).unwrap(), b"new");
        assert_eq!(std::fs::read(&there).unwrap(), b"old", "the other link must keep its bytes");
        let left: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(left.is_empty(), "no temporary file is left behind");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_new_file_and_its_missing_folder_are_made() {
        let dir = scratch("new");
        let path = dir.join("nested").join("a.txt");
        write_atomic(&path, b"whole").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"whole");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
