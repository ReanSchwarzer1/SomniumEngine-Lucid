# Feature reference

The longer list behind the README's summary. Everything here is in both Somnium Engine and Lucid unless it says
otherwise; what Lucid adds on top is listed in the [README](../README.md#what-lucid-adds). Where a feature has an
editor Help page, it is linked. Counts that were in older versions of this list have been left out where they were no
longer current; `tools/census/generate.py` gives today's numbers.

## Rendering

- Visibility-buffer pipeline with programmable vertex pulling and bindless resources (single global bind group).
- GPU-driven rendering: `multi_draw_indirect`, compute frustum culling, meshlet clusters, Hi-Z two-phase occlusion
  culling.
- Physically based shading (Cook-Torrance GGX) with an alternative cel-shading mode, plus multiple-scattering energy
  compensation on both the direct lobe and the IBL, Hammon's rough diffuse, micro-shadowing, and geometric specular
  antialiasing with a roughness-scaled bound on every lobe, the sun and the moon both on it.
- Two-sided foliage transport: a wrapped, view-scattered transmission lobe so a backlit leaf glows, on the same
  `shadow_factor` the reflected term uses. Occlusion composes rather than replaces: GTAO, the authored map, baked sky
  visibility and terrain layer occlusion multiply into one number, and micro-shadowing takes a material-only copy.
- Shadows: cascaded shadow maps with PCSS and contact shadows, plus sparse virtual shadow maps (clipmap page table,
  persistent physical atlas, per-page raster). Cascades stop at 100 m, so terrain also carries a baked heightfield
  horizon map that shadows it at any distance, cross-faded in where the last cascade fades out.
- Global illumination: ReSTIR DI/GI on wgpu acceleration structures, with a portable DDGI tier (SDF-traced 4×4×4
  volume, budgeted temporal SH updates) for hardware without ray query.
- Atmosphere and sky: Hillaire scattering LUTs driving both sky and IBL, analytic NOAA sun position, a day cycle, and
  volumetric clouds (Perlin–Worley, quarter-res adaptive march, cloud shadows folded into one `shadow_factor` every
  surface reads).
- Weather: coverage → precipitation → wetness on Lagarde's two time constants, porosity as a material channel, rain
  ripples on water.
- HDR pipeline: `Rgba16Float`, AgX/ACES tone mapping, GTAO, bloom, DoF, motion blur, deferred decals.
- Anti-aliasing as one authored choice: Off / FXAA / SMAA 1x / SMAA T2x / TAA / FSR 3, plus SMAA quality presets. (It
  was three independent booleans until MORROWIND-AC, where the default configuration was measured to run no AA pass
  while showing a checked box.)
- FSR 3 frame generation: AMD's FidelityFX 1.1.4 frame interpolation and optical flow translated to WGSL
  (`third_party/wgpu-ffx-fg`, 18 passes). It is paced by split submits, and the game UI is drawn fresh over real and
  generated frames alike. In the debug build, 42 fps became 77 presented when it was measured.
- Real-time path tracer: one path a pixel, starting from the raster's own visibility-buffer surface. Next-event
  estimation for the sun, the moon and resampled lamps; GGX visible-normal sampling; a reprojected temporal pass and an
  edge-stopping à-trous denoiser. A second mode traces reflections on glossy surfaces only, lit by the same lamps, in
  place of the environment specular.
- Spot-light cookies: every spot cone in every lighting path (shading, glass, fog, GI, the path tracer, scene capture)
  goes through one procedural pattern function: a torch reflector, blinds, window panes, a caged lamp, a grimy fixture.
- Skinned characters in ray tracing: posed meshes get their own BLAS, rebuilt within a reach, so characters cast traced
  lamp shadows and appear in reflections.
- Cached shadow cascades: revisions per cascade; a caster's movement under four of a cascade's texels leaves it cached,
  and posed casters reach only the cascades their shadow can land in.
- Lens flare from the threshold-free bloom chain; scene capture (a cheap second view for mirrors and screens); a
  PS2-style picture and animated picture shifts.
- A game's own graphics budget: per-feature choices laid over every scene in place of a quality tier, plus requests for
  what no scene authors (path tracing, traced reflections, lens flare) and runtime V-sync and frame-cap changes.
- Imported textures as BC7, only for the materials a level draws, with a content-keyed cache outside the repository;
  scene restore decodes in parallel and pipelines decode with upload.
- Order-independent transparency: weighted-blended, authored per scene, with the sorted path retained.
- Water: three-cascade 1024² inverse-FFT ocean spectrum with Jacobian whitecaps, temporal foam, Beer's-law transport, a
  shoreline SDF and a depth-faded shore. Reflections blend screen-space tracing, half-res hardware ray tracing, and the
  environment cube. See [Water](editor/water.md).
- Terrain: 32-layer material with strongest-four blending, triplanar cliffs, hex tiling and parallax occlusion; nested
  material clipmaps; virtual texturing streaming BC7 source pages into a fixed 64 MiB atlas. See
  [Terrain](editor/terrain.md).
- Terrain lighting: three heightfield bakes: horizon angles in eight azimuths for long-range sun shadow,
  cosine-weighted sky visibility with a bent direction so valleys are darker than the ridges above them, and a
  mip-chained relief normal that carries its own discarded variance into roughness.

## World and content

- glTF 2.0 loading (meshes, PBR materials, textures, skins), importable at runtime.
- Deterministic asset cook: versioned, integrity-checked payloads with a SHA-256 recipe key; changing a texture
  recooks its material's reverse closure while unrelated meshes stay cached.
- Residency: typed placeholders returned immediately, resolver I/O through the job system, per-frame upload budget,
  deterministic LRU, atomic publish.
- World partition: double-precision cell hashing, camera/player/volume want-state, transactional unload that persists
  real ECS components through the schema serializer.
- Cell streaming of read-only world cells built offline, and terrain windows whose geometry is resident only near the
  camera. See [Open worlds](editor/open_world.md). Lucid's version goes much further (see the README).
- Voxel world: 32³ padded chunks, `block_mesh` face culling, 3 LOD levels, sparse edit overlay.
- Heightmap terrain: chunked CDLOD-style LOD with crack-free stitching, real-time sculpting and texture painting.
- Foliage painting: brush over terrain with density, size and single-instance placement, from a CC0 palette. Entries
  that require a terrain layer underneath them say so in the log when they refuse, and the requirement is a brush
  setting rather than a fixed property of the entry.

## Scripting and simulation

- Luau (`mlua`, interpreter only): sandboxed, with a deferred command buffer as the only route into the world,
  generated type declarations, and acceptance budgets asserted as tests. See [Scripting](editor/scripting.md).
- Animation: skeletons, four-weight skinning, clips, `Blend1D`/`Blend2D`, masked layers, typed parameters and
  triggers, sync tracks, state machines with a bounded pose cache.
- Jobs: declared priorities and deadlines, cancellation, a budgeted main-thread completion drain, and profiler
  telemetry. A job whose deadline passed while queued is dropped rather than run.
- Jolt physics integration; Kira audio with buses, attenuation curves, cones, occlusion and Doppler.
- Binaural audio through the MIT KEMAR head responses (368 directions), and room acoustics: a reverb send that follows
  how enclosed the listener is, and a muffled track for occluded sounds.
- CPU particle system with GPU billboard instancing, and textured sprite and flipbook effects. See
  [Particles](editor/particles.md).
- Navigation (tiled navmesh, pathfinding, agents), behaviour graphs, a combat sandbox and a story/dialogue layer. See
  [Combat](editor/combat.md) and [Story](editor/story.md).

## Editor

- Schema-generated Details: editable fields across the registered component schemas are generated from the schema,
  with scoped undo, drag coalescing, multi-select intersection and mixed-value rows. A generated census table checks
  that no hand-written inspector is left.
- One command registry (more than 140 commands) generates the menus, Create surfaces, toolbar, tooltips, shortcuts,
  palette and F1 index.
- Node graph editor and timeline, both archetype-driven and feature-neutral, with material and animation catalogues as
  the first consumers.
- Curve and gradient editing as reflected values (`FieldType::Curve`), so a curve gets its Details row, undo and
  round-trip for free.
- Scene lifecycle: format-routed load, retained unknown components and fields (no silent data loss), header
  thumbnails, autosave and crash recovery, clickable undo history.
- Preferences with a `default → project.toml → editor.toml → SOMNIUM_*` order. Every `SOMNIUM_*` variable has a
  declared, checked route (schema field, setting, command or a stated harness reason), enforced by
  `tools/reachability/`.
- Transform gizmos, light gizmos, selection outline, infinite grid, camera bookmarks, statistics overlay, output log
  with severity chips and jump-to-source.
- Accessibility: roles, names, values, focus announcements, reduced motion, high contrast reusing the theme's certified
  contrast pairs.
- Projects and automation: open a game's own editor, drive the editor through semantic MCP controls
  (`tools/somnium_mcp`). See [Automation](editor/automation.md).

## Editor design system: Nocturne Atelier

- Colour pipeline: authored sRGB decodes to linear exactly once before the sRGB swapchain, straight alpha through the
  widget API. `#1C1E26` reaches the framebuffer as `#1C1E26`.
- Layered tokens: palette → semantic roles → component recipes → interaction state, so a widget asks for
  `style::button(state)` rather than picking a grey.
- Typography: Inter Regular/Medium/SemiBold and JetBrains Mono Regular/Medium behind named roles; numeric fields use
  the mono face so a scrub cannot shift a row.
- One icon family: Tabler outline icons plus original Somnium glyphs on the same 24×24 / 2 px grid, rasterized from
  SVG by `resvg`.
- The full contract is [`dev records/phase_26_Zeta.md`](../dev%20records/phase_26_Zeta.md).
