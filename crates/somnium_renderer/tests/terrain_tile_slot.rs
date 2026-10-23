//! A streamed terrain tile's slot: a sidecar read anywhere and adopted is the ground `load_binary` gives, a retired
//! slot hands every pool span back, and the terrain material pool says when it has no room for another slot.
#[path = "support/gpu.rs"]
mod gpu;

use glam::{Mat4, Vec3};
use somnium_renderer::geometry::GeometryPool;
use somnium_renderer::material::pool::TerrainMaterialPool;
use somnium_renderer::terrain::foliage_paint::PaintedFoliage;
use somnium_renderer::terrain::{self, TerrainData, TerrainDescriptor, UNALLOCATED, horizon, relief};

fn desc() -> TerrainDescriptor {
    // 2 x 2 chunks of 16 cells: 33 x 33 heights, 32 x 32 splat texels.
    TerrainDescriptor { chunk_cells: 16, grid_size: [2, 2], cell_size: 2.0, height_scale: 1.5, ..Default::default() }
}

#[test]
fn an_adopted_sidecar_is_the_ground_load_binary_gives_and_a_retired_slot_frees_its_spans() {
    // SAFETY: the only test in this binary, set before anything else runs. 256 keeps the procedural ground layers
    // (there are no packed ones in the folder below) small.
    unsafe { std::env::set_var("SOMNIUM_TERRAIN_RES", "256") };
    let (device, queue) = gpu::device();
    let dir = std::env::temp_dir().join(format!("terrain_tile_slot_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let assets = dir.join("no_layers");
    let make = |desc: TerrainDescriptor| TerrainData::new_with_asset_dir(&device, &queue, desc, false, &assets);

    // Noisy heights, splat and foliage, saved.
    let mut source = make(desc());
    let mut state = 0x9E37_79B9u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for height in &mut source.heightmap {
        *height = (next() % 4000) as f32 * 0.01;
    }
    for texel in &mut source.splatmap.data {
        for weight in texel.iter_mut() {
            *weight = next() as u8;
        }
    }
    source.painted_foliage = (0..24)
        .map(|i| PaintedFoliage {
            kind: i as u8,
            position: Vec3::new((next() % 64) as f32, (next() % 40) as f32 * 0.5, (next() % 64) as f32),
            yaw: (next() % 628) as f32 * 0.01,
            tilt: 0.0,
            scale: 0.5 + (next() % 100) as f32 * 0.01,
        })
        .collect();
    let file = dir.join("tile.scene.json.terrain0.bin");
    let path = file.to_str().unwrap();
    source.save_binary(path).unwrap();

    // load_binary into one terrain; read_sidecar and adopt_sidecar into another.
    let mut loaded = make(desc());
    let mut adopted = make(desc());
    for chunk in loaded.chunks.iter_mut().chain(adopted.chunks.iter_mut()) {
        chunk.dirty = false;
    }
    let (loaded_revision, adopted_revision) = (loaded.edit_revision, adopted.edit_revision);
    loaded.load_binary(path).unwrap();
    let dims = adopted.sidecar_dims();
    assert_eq!(dims, [33, 33, 32, 32]);
    let sidecar = terrain::read_sidecar(path, dims).unwrap();
    let model = Mat4::from_translation(Vec3::new(1048.0, 3.0, -2000.0));
    adopted.adopt_sidecar(&queue, sidecar, model, 800.0).unwrap();
    assert_eq!(loaded.heightmap, source.heightmap);
    assert_eq!(adopted.heightmap, source.heightmap);
    assert_eq!(loaded.splatmap.data, source.splatmap.data);
    assert_eq!(adopted.splatmap.data, source.splatmap.data);
    assert_eq!(loaded.painted_foliage, source.painted_foliage);
    assert_eq!(adopted.painted_foliage, source.painted_foliage);
    assert_eq!(loaded.edit_revision, loaded_revision, "load_binary leaves the revision alone");
    assert_eq!(adopted.edit_revision, adopted_revision + 1, "adopting is an edit");
    assert!(loaded.chunks.iter().chain(&adopted.chunks).all(|c| c.dirty), "every chunk rebuilds");
    assert_eq!(loaded.splatmap.dirty, Some((0, 0, 31, 31)));
    assert_eq!(adopted.splatmap.dirty, Some((0, 0, 31, 31)));
    assert_eq!(adopted.model, model);
    assert_eq!(adopted.geometry_stream_distance, 800.0);

    // The dimensions are checked before the foliage is read: a wrong size names itself even over corrupt foliage.
    std::fs::write(format!("{path}.foliage.json"), b"{ not foliage").unwrap();
    let mut wider = make(TerrainDescriptor { grid_size: [3, 2], ..desc() });
    let from_load = wider.load_binary(path).unwrap_err();
    let from_read = terrain::read_sidecar(path, wider.sidecar_dims()).err().unwrap();
    assert_eq!(from_load, "terrain sidecar dimensions do not match descriptor");
    assert_eq!(from_read, from_load);
    assert!(terrain::read_sidecar(path, dims).is_err(), "the right size then reads the corrupt foliage");
    std::fs::remove_file(format!("{path}.foliage.json")).unwrap();

    // bake() is the horizon and relief bake of the same heights.
    let mut sidecar = terrain::read_sidecar(path, dims).unwrap();
    assert!(sidecar.foliage.is_empty());
    sidecar.bake(desc().cell_size, desc().height_scale);
    let (horizon_maps, relief_maps) = sidecar.baked.as_ref().unwrap();
    let h = horizon::bake(&source.heightmap, 33, 33, 2.0, 1.5, horizon::HORIZON_SIZE);
    let r = relief::bake(&source.heightmap, 33, 33, 2.0, 1.5, relief::RELIEF_SIZE);
    assert_eq!(
        (&horizon_maps.angles_a, &horizon_maps.angles_b, &horizon_maps.sky, horizon_maps.size),
        (&h.angles_a, &h.angles_b, &h.sky, h.size)
    );
    assert_eq!((&relief_maps.levels, relief_maps.size), (&r.levels, r.size));
    // A baked sidecar adopts too, and its maps are not baked a second time on the next rebuild.
    let mut baked = make(desc());
    baked.adopt_sidecar(&queue, sidecar, Mat4::IDENTITY, 800.0).unwrap();
    assert!(!baked.horizon_dirty);
    assert!(adopted.horizon_dirty, "an unbaked sidecar rebakes on the next rebuild");

    // Retiring gives back exactly the resident chunks and every span, and keeps the heights.
    let mut pool = GeometryPool::new(&device);
    let baseline = pool.reserved_span_counts();
    adopted.reserve_pool_spans(&mut pool);
    adopted.ensure_index_blocks(&queue, &mut pool);
    let resident: Vec<u32> = adopted.chunks.iter().map(|c| c.vertex_offset).collect();
    assert!(resident.iter().all(|&offset| offset != UNALLOCATED));
    assert_ne!(pool.reserved_span_counts(), baseline);
    assert_eq!(adopted.retire(&mut pool), resident);
    assert_eq!(pool.reserved_span_counts(), baseline);
    assert!(adopted.chunks.iter().all(|c| c.vertex_offset == UNALLOCATED));
    assert!(adopted.painted_foliage.is_empty());
    assert_eq!(adopted.heightmap, source.heightmap);
    assert!(adopted.retire(&mut pool).is_empty(), "a retired slot has nothing left to give");

    // The material pool says when it has no entry left for another terrain, so a tile can be refused one rather than
    // share entry 0 with another terrain's ground.
    // SAFETY: as above; nothing else in this binary reads it.
    unsafe { std::env::set_var("SOMNIUM_TERRAIN_MATERIALS", "3") };
    let mut materials = TerrainMaterialPool::new(&device);
    assert_eq!(materials.room(), 3);
    assert_eq!((materials.allocate(), materials.room()), (Some(0), 2));
    assert_eq!((materials.allocate(), materials.allocate()), (Some(1), Some(2)));
    assert_eq!((materials.room(), materials.allocate()), (0, None), "full: no room, and no entry");
    materials.reset();
    assert_eq!(materials.room(), 3, "a scene reset gives every entry back");

    let _ = std::fs::remove_dir_all(&dir);
}
