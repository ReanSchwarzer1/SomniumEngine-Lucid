//! The ground's bare share near the eye (Sunscar, 19 Nov 2026), for the GPU grass proof: `SOMNIUM_COVER_CENSUS=1`.
//!
//! Every 30 frames it counts the pixels within 10 m and within 10-30 m of the eye as terrain (bare ground), GPU grass,
//! ground cover (cut-out foliage below the eye less 0.3 m) or other, and logs them with the eye:
//! `cover census: eye (x, y, z), blades on; 0-10 m bare 12.3% (terrain t, grass g, cover c, other o); 10-30 m ...`.
//! The bare share is terrain / (terrain + grass + cover). Off by default; no window shows it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

const BINS: usize = 8;
const BYTES: u64 = (BINS * 4) as u64;
const EVERY: u32 = 30;
const IDLE: u8 = 0;
const COPIED: u8 = 1;
const MAPPING: u8 = 2;
const READY: u8 = 3;

pub struct CoverCensus {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    counts: wgpu::Buffer,
    readback: wgpu::Buffer,
    params: wgpu::Buffer,
    state: Arc<AtomicU8>,
    frame: u32,
    eye: glam::Vec3,
    blades: bool,
}

impl CoverCensus {
    /// `None` unless `SOMNIUM_COVER_CENSUS=1`.
    pub fn new(
        device: &wgpu::Device,
        shaders: &crate::shaders::Shaders,
        global_layout: &wgpu::BindGroupLayout,
    ) -> Option<Self> {
        if std::env::var("SOMNIUM_COVER_CENSUS").as_deref() != Ok("1") {
            return None;
        }
        let compute = wgpu::ShaderStages::COMPUTE;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cover Census BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: compute,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: compute,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cover Census Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders.source_or_panic("cover_census.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cover Census Pipeline Layout"),
            bind_group_layouts: &[Some(global_layout), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cover Census Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cover Census Counts"),
            size: BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cover Census Readback"),
            size: BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cover Census Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        tracing::info!("cover census on: the bare share within 10 m and 30 m, every {EVERY} frames");
        Some(Self { pipeline, layout, counts, readback, params, state: Arc::new(AtomicU8::new(IDLE)), frame: 0, eye: glam::Vec3::ZERO, blades: false })
    }

    /// Count this frame's finished visibility buffer, every 30 frames while no read is pending.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        global_group: &wgpu::BindGroup,
        vis_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        size: (u32, u32),
        eye: glam::Vec3,
        blades: bool,
    ) {
        self.frame = self.frame.wrapping_add(1);
        if self.frame % EVERY != 0 || self.state.load(Ordering::Acquire) != IDLE || size.0 == 0 || size.1 == 0 {
            return;
        }
        queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&[size.0, size.1, 0u32, 0u32]));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cover Census BG"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(vis_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: self.counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.params.as_entire_binding() },
            ],
        });
        encoder.clear_buffer(&self.counts, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cover Census"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, global_group, &[]);
            pass.set_bind_group(1, &group, &[]);
            pass.dispatch_workgroups(size.0.div_ceil(16), size.1.div_ceil(16), 1);
        }
        encoder.copy_buffer_to_buffer(&self.counts, 0, &self.readback, 0, BYTES);
        self.eye = eye;
        self.blades = blades;
        self.state.store(COPIED, Ordering::Release);
    }

    /// Start reading the copy `record` made. After the submit that holds it.
    pub fn after_submit(&mut self) {
        if self.state.load(Ordering::Acquire) != COPIED {
            return;
        }
        self.state.store(MAPPING, Ordering::Release);
        let state = Arc::clone(&self.state);
        self.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            state.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
        });
    }

    /// Log a finished read. Once a frame, before `after_submit`.
    pub fn collect(&mut self) {
        if self.state.load(Ordering::Acquire) != READY {
            return;
        }
        let mut c = [0u32; BINS];
        {
            let view = self.readback.slice(..).get_mapped_range().expect("read only once its map fired");
            for (i, chunk) in view.chunks_exact(4).take(BINS).enumerate() {
                c[i] = u32::from_le_bytes(chunk.try_into().expect("4 bytes"));
            }
        }
        self.readback.unmap();
        self.state.store(IDLE, Ordering::Release);
        let e = self.eye;
        tracing::info!(
            "cover census: eye ({:.1}, {:.1}, {:.1}), blades {}; 0-10 m {}; 10-30 m {}",
            e.x,
            e.y,
            e.z,
            if self.blades { "on" } else { "off" },
            band(&c[0..4]),
            band(&c[4..8])
        );
    }
}

/// One band's line: bare share, then the counts (terrain, grass, cover, other).
fn band(c: &[u32]) -> String {
    let ground = c[0] + c[1] + c[2];
    let bare = if ground == 0 { 0.0 } else { 100.0 * c[0] as f32 / ground as f32 };
    format!("bare {bare:.1}% (terrain {}, grass {}, cover {}, other {})", c[0], c[1], c[2], c[3])
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_bare_share_is_terrain_among_ground_pixels() {
        assert_eq!(super::band(&[25, 50, 25, 900]), "bare 25.0% (terrain 25, grass 50, cover 25, other 900)");
        assert_eq!(super::band(&[0, 0, 0, 7]), "bare 0.0% (terrain 0, grass 0, cover 0, other 7)");
    }
}
