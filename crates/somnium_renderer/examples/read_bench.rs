//! Times the CPU half of streaming for a list of glTF files, as a streamed world's cell readers do it: each file
//! decoded (`somnium_asset::load_gltf`) and prepared (`SomniumRenderer::prepare_scene_for`) on one of `threads`
//! threads. No GPU. The list is one path a line, repeats included (a file read by two cells is read twice).
//!
//!     cargo run --release -p somnium_renderer --example read_bench -- <list> [threads] [MiB of decoded images shared]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let list = std::fs::read_to_string(args.next().expect("a file of glTF paths, one a line")).expect("the list reads");
    let files: Vec<&str> = list.lines().map(str::trim).filter(|line| !line.is_empty()).collect();
    let threads: usize = args.next().and_then(|t| t.parse().ok()).unwrap_or(4);
    let shared: usize = args.next().and_then(|t| t.parse().ok()).unwrap_or(0);
    somnium_asset::share_decoded_images(shared << 20);
    let (next, unread) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let (load_us, prepare_us, texels, textures) =
        (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                while let Some(file) = files.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let reading = Instant::now();
                    let Ok(scene) = somnium_asset::load_gltf(file) else {
                        unread.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    load_us.fetch_add(reading.elapsed().as_micros() as u64, Ordering::Relaxed);
                    let preparing = Instant::now();
                    let prepared = somnium_renderer::SomniumRenderer::prepare_scene_for(&scene, None, true);
                    prepare_us.fetch_add(preparing.elapsed().as_micros() as u64, Ordering::Relaxed);
                    let real = scene.textures.iter().filter(|t| t.width > 1);
                    textures.fetch_add(real.clone().count() as u64, Ordering::Relaxed);
                    texels.fetch_add(real.map(|t| u64::from(t.width) * u64::from(t.height)).sum(), Ordering::Relaxed);
                    drop(prepared);
                }
            });
        }
    });
    let seconds = |us: &AtomicU64| us.load(Ordering::Relaxed) as f64 / 1e6;
    println!(
        "{} files ({} unread) on {threads} threads, {shared} MiB shared: {:.1} s; decode {:.1} thread-s, prepare {:.1} thread-s; {} textures, {:.0} Mpx",
        files.len(),
        unread.load(Ordering::Relaxed),
        started.elapsed().as_secs_f64(),
        seconds(&load_us),
        seconds(&prepare_us),
        textures.load(Ordering::Relaxed),
        texels.load(Ordering::Relaxed) as f64 / 1e6,
    );
}
