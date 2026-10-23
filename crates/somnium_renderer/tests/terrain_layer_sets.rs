//! A streamed tile's slot wears any region's ground layers: two slots of two sets show different layer arrays, a slot
//! switched to another set and back shows the first set's arrays again, and a set no slot wears any more is freed.
#[path = "support/gpu.rs"]
mod gpu;

use std::sync::Arc;

use somnium_renderer::geometry::GeometryPool;
use somnium_renderer::terrain::textures::{LAYER_MATERIALS, LAYER_TILING, TERRAIN_HERO_LAYERS};
use somnium_renderer::terrain::{TerrainData, TerrainDescriptor};

fn desc() -> TerrainDescriptor {
    TerrainDescriptor { chunk_cells: 16, grid_size: [2, 2], cell_size: 2.0, ..Default::default() }
}

/// The texture behind a terrain's single-layer albedo view of `layer`, as its bindless slot would hold it.
fn albedo(terrain: &TerrainData, layer: usize) -> wgpu::Texture {
    terrain.layer_views(layer).expect("a resident layer").0.texture().clone()
}

/// The texture behind its surface view of `layer`.
fn surface(terrain: &TerrainData, layer: usize) -> wgpu::Texture {
    terrain.layer_views(layer).expect("a resident layer").1.texture().clone()
}

#[test]
fn a_slot_wears_any_sets_ground_and_a_set_no_slot_wears_is_freed() {
    // SAFETY: the only test in this binary, set before anything else runs. 256 keeps the procedural ground layers
    // (the folders below hold no packed ones) small.
    unsafe { std::env::set_var("SOMNIUM_TERRAIN_RES", "256") };
    let (device, queue) = gpu::device();
    let dir = std::env::temp_dir().join(format!("terrain_layer_sets_{}", std::process::id()));
    let tyffir = dir.join("terrain");
    let verglas = tyffir.join("sets").join("verglas");
    std::fs::create_dir_all(&verglas).unwrap();
    // Verglas's first layer is packed from a texture 8 m wide; Tyffir keeps the shipped widths.
    std::fs::write(verglas.join("layer_tiling.json"), format!("{{\"{}\": 8.0}}", LAYER_MATERIALS[0])).unwrap();
    let make = |assets: &std::path::Path| TerrainData::new_with_asset_dir(&device, &queue, desc(), false, assets);
    let extra = TERRAIN_HERO_LAYERS as usize + 3;

    // Two slots of two sets: different arrays in both banks, and each set's own tiling.
    let mut a = make(&tyffir);
    let mut b = make(&verglas);
    assert_eq!((a.layer_dir.as_path(), b.layer_dir.as_path()), (tyffir.as_path(), verglas.as_path()));
    assert!(!Arc::ptr_eq(&a.layer_textures, &b.layer_textures));
    for layer in [0, 7, extra] {
        assert_ne!(albedo(&a, layer), albedo(&b, layer), "layer {layer}");
        assert_ne!(surface(&a, layer), surface(&b, layer), "layer {layer}");
    }
    assert_eq!(a.layers[0].tiling, LAYER_TILING[0]);
    assert_eq!(b.layers[0].tiling, 1.0 / 8.0);
    assert_eq!(b.layers[1].tiling, LAYER_TILING[1], "an unnamed layer keeps the shipped width");
    // A third slot of Tyffir's shares A's arrays rather than loading its own.
    let c = make(&tyffir);
    assert!(Arc::ptr_eq(&a.layer_textures, &c.layer_textures));

    // A switches to Verglas and back: Verglas's arrays and tiling, then Tyffir's again (C kept them).
    let tyffir_albedo = albedo(&a, 0);
    let mut pool = GeometryPool::new(&device);
    a.reserve_pool_spans(&mut pool);
    a.rebuild_dirty_chunks(&queue, &mut pool, &mut Vec::new());
    assert!(!a.macro_dirty(), "a rebuild makes the macro map");
    assert!(a.set_layer_dir(&device, &queue, false, &verglas));
    assert!(Arc::ptr_eq(&a.layer_textures, &b.layer_textures), "a set worn already is shared, not loaded again");
    assert_eq!((albedo(&a, extra), surface(&a, 0)), (albedo(&b, extra), surface(&b, 0)));
    assert_eq!(a.layers[0].tiling, 1.0 / 8.0);
    assert!(a.macro_dirty(), "the macro map is made again from the new layers' colours");
    assert_eq!(a.layer_dir, verglas);
    assert!(a.set_layer_dir(&device, &queue, false, &tyffir));
    assert_eq!(albedo(&a, 0), tyffir_albedo, "switched back: the first set's views");
    assert_eq!(surface(&a, extra), surface(&c, extra));
    assert_eq!(a.layers[0].tiling, LAYER_TILING[0]);

    // B, the last slot wearing Verglas, takes Tyffir's ground: nothing holds Verglas's arrays any more.
    let verglas_set = Arc::downgrade(&b.layer_textures);
    let verglas_albedo = albedo(&b, 0);
    assert_eq!(verglas_set.strong_count(), 1);
    assert!(b.set_layer_dir(&device, &queue, false, &tyffir));
    assert_eq!(verglas_set.strong_count(), 0, "a set no slot wears is freed");
    assert!(Arc::ptr_eq(&b.layer_textures, &a.layer_textures));
    // Worn again, it is loaded again: new arrays, its tiling as before.
    assert!(b.set_layer_dir(&device, &queue, false, &verglas));
    assert_ne!(albedo(&b, 0), verglas_albedo, "loaded afresh, not kept");
    assert_ne!(albedo(&b, 0), tyffir_albedo);
    assert_eq!(b.layers[0].tiling, 1.0 / 8.0);
    // The set worn already changes nothing.
    let worn = Arc::clone(&b.layer_textures);
    assert!(b.set_layer_dir(&device, &queue, false, &verglas));
    assert!(Arc::ptr_eq(&worn, &b.layer_textures));

    drop((a, b, c, worn, pool));
    let _ = std::fs::remove_dir_all(&dir);
}
