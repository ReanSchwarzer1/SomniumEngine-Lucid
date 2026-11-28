//! Check glTF files as the engine reads them, before they are committed (Sunscar's rule of 6 Nov 2026, 20:05: "validate
//! every glTF you write before you commit it", after 84 mismatched paving files and a lost GPU device): each buffer's
//! `byteLength` against its `.bin`, the textures it names, then every primitive read as the engine's importer reads it
//! (accessors that read, indices in range, corners that are numbers). `--full` loads each file through the importer
//! itself, images and all (a quarter of a second a file, where the quick check takes milliseconds).
//!
//! `--open-edges` also names every piece drawn on one side only whose mesh has open edges, as the importer warns of
//! them (`somnium_asset::open_edges`): where the sky shows through as a thin line, as under the roofs the developer saw
//! on 27 Nov 2026. A cliff face or a rock with its foot in the ground is open on purpose: these are not broken.
//!
//! ```text
//! gltf_check [--full] [--open-edges] <file or folder>...      every .gltf and .glb in each folder, and below it
//! ```
//!
//! One `BROKEN <file>: <why>` line per broken file, one `OPEN <file>: <piece> (<material>) <open edges> of <triangles>
//! triangles` line per open piece, then a count. Exits 1 when any is broken, 2 when given nothing.

use std::path::{Path, PathBuf};

fn main() {
    let mut args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let full = args.iter().any(|a| a.as_os_str() == "--full");
    let open_edges = args.iter().any(|a| a.as_os_str() == "--open-edges");
    args.retain(|a| a.as_os_str() != "--full" && a.as_os_str() != "--open-edges");
    if args.is_empty() {
        eprintln!(
            "usage: gltf_check [--full] [--open-edges] <file or folder>...  (every .gltf and .glb in a folder and below it)"
        );
        std::process::exit(2);
    }
    let mut files = Vec::new();
    for path in &args {
        collect(path, &mut files);
    }
    files.sort();
    files.dedup();
    let started = std::time::Instant::now();
    let mut broken = 0usize;
    let (mut open_files, mut open_pieces, mut open_time) = (0usize, 0usize, std::time::Duration::ZERO);
    for file in &files {
        let mut problems = somnium_asset::gltf_problems(file);
        if problems.is_empty() && full {
            if let Err(error) = somnium_asset::load_gltf(file) {
                problems.push(format!("the engine's importer refuses it: {error}"));
            }
        }
        if !problems.is_empty() {
            broken += 1;
            println!("BROKEN {}: {}", file.display(), problems.join("; "));
        } else if open_edges {
            let looked = std::time::Instant::now();
            let pieces = somnium_asset::open_edge_pieces(file).unwrap_or_default();
            open_time += looked.elapsed();
            open_files += usize::from(!pieces.is_empty());
            open_pieces += pieces.len();
            for piece in pieces {
                println!(
                    "OPEN {}: {} ({}) {} of {} triangles",
                    file.display(),
                    piece.name,
                    piece.material,
                    piece.open_edges,
                    piece.triangles
                );
            }
        }
    }
    if open_edges {
        println!(
            "{open_pieces} one-sided pieces with open edges in {open_files} files ({:.2} s of reading and counting)",
            open_time.as_secs_f32()
        );
    }
    println!("{broken} of {} glTF files broken ({:.1} s)", files.len(), started.elapsed().as_secs_f32());
    std::process::exit(i32::from(broken > 0));
}

/// Every glTF under `path` (a folder, recursively), or `path` itself (a missing file is kept, so it is reported).
fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else { return };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                collect(&child, out);
            } else if is_gltf(&child) {
                out.push(child);
            }
        }
    } else {
        out.push(path.to_path_buf());
    }
}

fn is_gltf(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gltf") || e.eq_ignore_ascii_case("glb"))
}
