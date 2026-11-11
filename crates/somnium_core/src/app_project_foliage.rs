/// A palette source read off the main thread: its glTF decoded, its textures and meshes made ready to upload.
pub(crate) type PaletteRead = Result<(somnium_asset::LoadedScene, somnium_renderer::PreparedScene), String>;

// Project palette adapter: native terrain instances share one upload per source.
impl<G: GameApp> Engine<G> {
    /// Whether a palette source is uploaded (`None`: it failed). One that is not is read off the main thread, as a
    /// cell's sources are: at most as many at once as cells are read, and a later frame asks again.
    ///
    /// A streamed world's first frame loaded every palette source it drew, one after another, on the main thread: at
    /// Monargent that frame took over a minute after the first fill's ninety seconds, and each tile that came into
    /// reach afterwards stalled a frame for its new kinds (Sunscar, 12 Nov 2026).
    fn palette_source(&mut self, source: &std::path::Path, leaves_by_material: bool) -> Option<bool> {
        if self.project_foliage_sources.contains_key(source) {
            return Some(true);
        }
        if self.palette_failed.contains(source) {
            return None;
        }
        let readers = if self.cell_streamer.filling { crate::cell_streaming::fill_readers() } else { crate::cell_streaming::READERS };
        if !self.palette_reads.contains_key(source) && self.palette_reads.len() < readers {
            let path = source.to_path_buf();
            let compressing =
                self.render_ctx.as_ref().is_some_and(somnium_renderer::SomniumRenderer::imported_compression);
            somnium_asset::share_decoded_images(crate::cell_streaming::image_share_bytes());
            let gate = self.cell_streamer.gate.clone();
            let (send, receive) = std::sync::mpsc::sync_channel(1);
            std::thread::spawn(move || {
                // A turn at the cells' gate: the palette and the cells read as many at once between them.
                let _turn = gate.take();
                let read = somnium_asset::load_gltf(&path).map_err(|error| error.to_string()).map(|scene| {
                    let prepared = somnium_renderer::SomniumRenderer::prepare_scene_for(&scene, None, compressing);
                    (scene, prepared)
                });
                let _ = send.send(read);
            });
            self.palette_reads.insert(source.to_path_buf(), (receive, leaves_by_material));
        }
        Some(false)
    }

    /// Upload the palette sources whose reads have finished: at least one, then until `budget` of main-thread time
    /// is spent (`None`: all that are ready). True while any is still being read.
    pub(super) fn pump_palette_reads(&mut self, budget: Option<std::time::Duration>) -> bool {
        let started = std::time::Instant::now();
        let sources: Vec<std::path::PathBuf> = self.palette_reads.keys().cloned().collect();
        let mut uploaded = false;
        for source in sources {
            if uploaded && budget.is_some_and(|budget| started.elapsed() >= budget) {
                break;
            }
            let read = match self.palette_reads[&source].0.try_recv() {
                Ok(read) => read,
                Err(std::sync::mpsc::TryRecvError::Empty) => continue,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("the palette reader stopped".to_string()),
            };
            let leaves_by_material = self.palette_reads.remove(&source).is_some_and(|(_, leaves)| leaves);
            match (read, self.renderer.as_mut().zip(self.render_ctx.as_ref())) {
                (Ok((scene, prepared)), Some((renderer, ctx))) => {
                    let parts = renderer
                        .upload_prepared_scene_for(ctx, &scene, None, prepared)
                        .iter()
                        .map(|node| FoliagePart {
                            vertex_offset: node.vertex_offset,
                            index_offset: node.index_offset,
                            index_count: node.index_count,
                            material_id: node.material_id,
                            local: node.transform,
                            is_leaf: leaves_by_material
                                && scene.materials.get(node.material_index).is_some_and(|material| {
                                    material.foliage || material.alpha_mode != somnium_asset::AlphaMode::Opaque
                                }),
                        })
                        .collect();
                    self.project_foliage_sources.insert(source, parts);
                    crate::cell_streaming::free_elsewhere(scene);
                    uploaded = true;
                }
                (Ok(_), None) => {}
                (Err(error), _) => {
                    warn!(%error, source = %source.display(), "Project foliage import failed");
                    self.palette_failed.insert(source);
                }
            }
        }
        !self.palette_reads.is_empty()
    }

    /// Ask for the palette meshes the terrains in the world draw (their painted kinds, and the cover their ground set
    /// grows), as a frame's foliage pass does. Called while a first fill holds the frame, so the palette is read
    /// beside the cells and the first frame has its trees.
    pub(super) fn prefetch_foliage_palette(&mut self) {
        let terrains: Vec<(u32, String)> = self
            .world
            .entities_with::<TerrainComponent>()
            .filter_map(|e| {
                if crate::is_hidden(&self.world, e) || !self.world.get::<FoliageComponent>(e)?.enabled {
                    return None;
                }
                let layer_set = self
                    .world
                    .get::<crate::cell_streaming::TerrainStreamingComponent>(e)
                    .map(|streaming| streaming.layer_set.clone())
                    .unwrap_or_default();
                Some((self.world.get::<TerrainComponent>(e)?.terrain_id, layer_set))
            })
            .collect();
        for (terrain_id, layer_set) in terrains {
            let mut kinds = {
                let Some(terrain) = self.renderer.as_ref().and_then(|renderer| renderer.terrain(terrain_id)) else {
                    continue;
                };
                let grid = self.foliage_grids.entry(terrain_id).or_default();
                grid.refresh(terrain.edit_revision, &terrain.painted_foliage);
                grid.kinds.clone()
            };
            kinds.extend(
                self.project_foliage
                    .iter()
                    .filter(|(_, entry)| entry.ground_cover.as_ref().is_some_and(|cover| cover.grows_on(&layer_set)))
                    .map(|(kind, _)| *kind),
            );
            for kind in kinds {
                self.ensure_palette_mesh(kind);
            }
        }
    }
    fn foliage_kind_exists(&self, kind: u16) -> bool {
        (kind as usize) < FOLIAGE_PALETTE.len() || self.project_foliage.contains_key(&kind)
    }

    fn foliage_kind_name(&self, kind: u16) -> &str {
        self.project_foliage
            .get(&kind)
            .map(|e| e.name.as_str())
            .or_else(|| FOLIAGE_PALETTE.get(kind as usize).map(|e| e.name))
            .unwrap_or("Unknown foliage")
    }

    fn foliage_palette_labels(&self) -> Vec<(u16, String)> {
        FOLIAGE_PALETTE
            .iter()
            .enumerate()
            .map(|(i, e)| (i as u16, e.name.to_owned()))
            .chain(
                self.project_foliage
                    .iter()
                    .map(|(&id, e)| (id, e.name.clone())),
            )
            .collect()
    }

    fn ensure_project_palette_mesh(&mut self, kind: u16) {
        let index = kind as usize;
        // Past every table: no palette or sidecar can name it (both refuse it), so it is drawn as unavailable.
        if index >= FOLIAGE_KINDS || self.foliage_meshes[index].is_some() || self.foliage_failed[index] {
            return;
        }
        let Some(entry) = self.project_foliage.get(&kind).cloned() else {
            warn!(kind, "Painted foliage kind has no project palette entry");
            self.foliage_failed[index] = true;
            return;
        };
        // A streamed world reads the kind's sources off the main thread (`palette_source`): the kind is drawn once
        // its own mesh and its far forms are all in. Any other scene loads them here and now, as it always has.
        if self.cell_streamer.key.is_some() {
            let mut waiting = false;
            let wanted = [
                (Some(&entry.source), true),
                (entry.lod.as_ref().map(|lod| &lod.source), false),
                (entry.impostor.as_ref().map(|impostor| &impostor.source), false),
            ];
            for (source, own) in wanted {
                let Some(source) = source else { continue };
                match self.palette_source(source, own) {
                    Some(true) => {}
                    Some(false) => waiting = true,
                    // Its own mesh did not read: the kind is not drawn. A far form that did not read is left out
                    // below, with a warning, and the kind keeps its full geometry.
                    None if own => {
                        self.foliage_failed[index] = true;
                        return;
                    }
                    None => {}
                }
            }
            if waiting {
                return;
            }
        }
        if !self.project_foliage_sources.contains_key(&entry.source) {
            let Some((renderer, ctx)) = self.renderer.as_mut().zip(self.render_ctx.as_ref()) else {
                return;
            };
            let scene = match somnium_asset::load_gltf(&entry.source) {
                Ok(scene) => scene,
                Err(error) => {
                    warn!(%error, source=%entry.source.display(),"Project foliage import failed");
                    self.foliage_failed[index] = true;
                    return;
                }
            };
            // Project materials are authored content. Preserve their alpha,
            // transmission and roughness instead of applying legacy palette guesses.
            let uploaded = renderer.upload_scene(ctx, &scene);
            let parts = uploaded
                .iter()
                .map(|node| FoliagePart {
                    vertex_offset: node.vertex_offset,
                    index_offset: node.index_offset,
                    index_count: node.index_count,
                    material_id: node.material_id,
                    local: node.transform,
                    is_leaf: scene
                        .materials
                        .get(node.material_index)
                        .is_some_and(|material| {
                            material.foliage
                                || material.alpha_mode != somnium_asset::AlphaMode::Opaque
                        }),
                })
                .collect();
            self.project_foliage_sources
                .insert(entry.source.clone(), parts);
        }
        let source = &self.project_foliage_sources[&entry.source];
        let adjustment = entry
            .local_transform
            .map_or(glam::Mat4::IDENTITY, |m| glam::Mat4::from_cols_array(&m));
        let mut parts = Vec::with_capacity(entry.primitives.len());
        for ordinal in &entry.primitives {
            let Some(mut part) = source.get(*ordinal as usize).copied() else {
                warn!(
                    kind,
                    ordinal, "Project foliage primitive ordinal is outside its source"
                );
                self.foliage_failed[index] = true;
                return;
            };
            part.local = adjustment * part.local;
            parts.push(part);
        }
        info!(kind,name=%entry.name,parts=parts.len(),"Project foliage palette entry ready");
        self.foliage_meshes[index] = Some(parts);
        if let Some(lod) = &entry.lod {
            self.foliage_lod_meshes[index] = self.alternate_foliage_mesh(kind, lod, adjustment, "LOD");
        }
        if let Some(impostor) = &entry.impostor {
            self.foliage_impostor_meshes[index] = self.alternate_foliage_mesh(kind, impostor, adjustment, "impostor");
        }
    }

    /// A palette entry's LOD or impostor mesh: its source uploaded once, its primitives in order. `None` when it
    /// cannot be had; the entry then keeps drawing what it has.
    fn alternate_foliage_mesh(
        &mut self,
        kind: u16,
        alt: &crate::foliage_palette::ProjectLod,
        adjustment: glam::Mat4,
        what: &str,
    ) -> Option<Vec<FoliagePart>> {
        if !self.project_foliage_sources.contains_key(&alt.source) {
            let (renderer, ctx) = self.renderer.as_mut().zip(self.render_ctx.as_ref())?;
            match somnium_asset::load_gltf(&alt.source) {
                Ok(scene) => {
                    let nodes = renderer.upload_scene(ctx, &scene);
                    let parts = nodes
                        .iter()
                        .map(|node| FoliagePart {
                            vertex_offset: node.vertex_offset,
                            index_offset: node.index_offset,
                            index_count: node.index_count,
                            material_id: node.material_id,
                            local: node.transform,
                            is_leaf: false,
                        })
                        .collect();
                    self.project_foliage_sources.insert(alt.source.clone(), parts);
                }
                Err(error) => {
                    warn!(%error, kind, what, "Foliage alternate mesh import failed; retaining full geometry");
                    return None;
                }
            }
        }
        let source = &self.project_foliage_sources[&alt.source];
        let parts: Option<Vec<_>> = alt
            .primitives
            .iter()
            .map(|ordinal| {
                let mut part = *source.get(*ordinal as usize)?;
                part.local = adjustment * part.local;
                Some(part)
            })
            .collect();
        if parts.is_none() {
            warn!(kind, what, "Foliage alternate mesh ordinal missing; retaining full geometry");
        }
        parts
    }
}
