//! Lightning channels (Sunscar weather, 11 Nov 2026): the bolt itself, drawn as light.
//!
//! A [`Bolt`] is a channel between two world points. Its path is grown here the way a channel looks: straight over
//! kilometres, crooked over tens of metres (each span's middle is pushed aside by a share of its length, again and
//! again down to about 14 m), with forks that leave it at 25-45 degrees, run a third to a half as far and carry a
//! third of the light. The pass draws every span as a ribbon facing the eye, additive into the HDR scene after the
//! cloud composite and behind the scene's own depth. A channel is centimetres wide and far away, so the ribbon is
//! never thinner than a pixel and its radiance is the channel's light per metre spread over the width drawn: the
//! light that reaches the eye is the same at any distance, and bloom does the rest.
//!
//! The light a bolt throws is not here: the cloud takes it as a `StormFlash`, the land as an ordinary light.

use glam::Vec3;

/// A channel to draw this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bolt {
    pub from: Vec3,
    pub to: Vec3,
    /// Its path: the same seed grows the same channel, so a flash's return strokes relight one path.
    pub seed: u32,
    /// Luminous intensity per metre of channel, by colour, candela per metre (a return stroke about 2e7).
    pub candela_per_metre: Vec3,
    /// How much it forks: 0 a single channel, 1 heavily branched.
    pub branching: f32,
    /// The share of its length that exists yet, from `from`: a crawling discharge grows; a return stroke is whole (1).
    pub grown: f32,
}

/// One straight span of a channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub a: Vec3,
    pub b: Vec3,
    /// Its share of the main channel's light (forks carry less).
    pub level: f32,
    /// Where along the bolt it starts, 0..1 (for `Bolt::grown`).
    pub along: f32,
}

/// Spans a frame may draw: a heavily forked two-kilometre channel is about 600.
pub const MAX_SPANS: usize = 4096;
/// Below this length a span is drawn straight.
const FINEST: f32 = 14.0;

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        // xorshift32: small, and the same on every machine.
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        #[allow(clippy::cast_precision_loss)]
        let unit = (self.0 >> 8) as f32 / (1u32 << 24) as f32;
        unit
    }

    /// Roughly normal, -1..1 mostly: the mean of three.
    fn spread(&mut self) -> f32 {
        (self.next() + self.next() + self.next()) * (2.0 / 3.0) - 1.0
    }
}

/// A unit vector square to `dir`, turned at random about it.
fn aside(dir: Vec3, rng: &mut Rng) -> Vec3 {
    let u = dir.any_orthonormal_vector();
    let v = dir.cross(u);
    let (s, c) = (rng.next() * std::f32::consts::TAU).sin_cos();
    u * c + v * s
}

#[allow(clippy::too_many_arguments)]
fn channel(a: Vec3, b: Vec3, level: f32, from: f32, to: f32, branching: f32, forks_left: u32, rng: &mut Rng, out: &mut Vec<Span>) {
    let length = a.distance(b);
    if length < FINEST * 2.0 || out.len() >= MAX_SPANS {
        out.push(Span { a, b, level, along: from });
        return;
    }
    let dir = (b - a) / length;
    let side = aside(dir, rng);
    let middle = (a + b) * 0.5 + side * (length * 0.16 * rng.spread());
    let half = (from + to) * 0.5;
    channel(a, middle, level, from, half, branching, forks_left, rng, out);
    // A fork leaves the bend on its outer side, still heading mostly the channel's way.
    if forks_left > 0 && length > 60.0 && rng.next() < branching * 0.55 {
        let angle = (25.0 + 20.0 * rng.next()).to_radians();
        let way = (dir * angle.cos() + aside(dir, rng) * angle.sin()).normalize();
        let reach = length * (0.35 + 0.25 * rng.next());
        channel(middle, middle + way * reach, level * 0.35, half, half + (to - half) * 0.6, branching, forks_left - 1, rng, out);
    }
    channel(middle, b, level, half, to, branching, forks_left, rng, out);
}

impl Bolt {
    /// Its spans, in order along it (forks after the bend they leave).
    pub fn grow(&self, out: &mut Vec<Span>) {
        // A bolt with an end that is not a place is no bolt: its length is never under the finest span, so the
        // recursion below would not end.
        if !(self.from.is_finite() && self.to.is_finite()) {
            return;
        }
        let start = out.len();
        let mut rng = Rng(self.seed.max(1).wrapping_mul(0x9E37_79B9) | 1);
        channel(self.from, self.to, 1.0, 0.0, 1.0, self.branching.clamp(0.0, 1.0), 3, &mut rng, out);
        if self.grown < 1.0 {
            let grown = self.grown.max(0.0);
            let mut kept = start;
            for i in start..out.len() {
                if out[i].along <= grown {
                    out[kept] = out[i];
                    kept += 1;
                }
            }
            out.truncate(kept);
        }
    }
}

/// One corner of a span's ribbon, as `lightning.wgsl` reads it.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Corner {
    a: [f32; 3],
    /// 0 this corner is at `a`, 1 at `b`.
    end: f32,
    b: [f32; 3],
    /// -1 or 1: which side of the channel.
    side: f32,
    /// Candela per metre, by colour, with the span's level in.
    light: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    /// Camera-relative, unjittered.
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    /// Radians a pixel spans, at the middle of the picture.
    pixel_angle: f32,
    viewport: [f32; 2],
    _pad: [f32; 2],
}

pub struct LightningPass {
    /// The channels to draw this frame. A game sets it; empty draws nothing and costs nothing.
    pub bolts: Vec<Bolt>,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    corners: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    spans: Vec<Span>,
    count: u32,
}

impl LightningPass {
    pub fn new(device: &wgpu::Device, shaders: &crate::shaders::Shaders, target: wgpu::TextureFormat) -> Self {
        let source = shaders.source_or_panic("lightning.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lightning.wgsl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lightning BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lightning PL"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lightning"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Corner>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target,
                    // Light adds.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            bolts: Vec::new(),
            pipeline,
            layout,
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Lightning Params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            corners: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Lightning Corners"),
                size: (MAX_SPANS * 6 * std::mem::size_of::<Corner>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bind_group: None,
            spans: Vec::new(),
            count: 0,
        }
    }

    /// Grow this frame's channels and upload their ribbons. `view_proj` is camera-relative and unjittered (a
    /// perspective's). Nothing is written when there are no bolts.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth: &wgpu::TextureView,
        size: (u32, u32),
        view_proj: glam::Mat4,
        camera_pos: Vec3,
    ) {
        self.count = 0;
        if self.bolts.is_empty() {
            return;
        }
        // Built for this frame's depth every frame that has a bolt: the scene's depth can be rebuilt at the same size.
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lightning BG"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
            ],
        }));
        let mut corners = Vec::new();
        for bolt in &self.bolts {
            self.spans.clear();
            bolt.grow(&mut self.spans);
            for span in &self.spans {
                if corners.len() / 6 >= MAX_SPANS {
                    break;
                }
                let light = (bolt.candela_per_metre * span.level).to_array();
                for (end, side) in [(0.0, -1.0), (0.0, 1.0), (1.0, -1.0), (1.0, -1.0), (0.0, 1.0), (1.0, 1.0)] {
                    corners.push(Corner { a: span.a.to_array(), end, b: span.b.to_array(), side, light, _pad: 0.0 });
                }
            }
        }
        if corners.is_empty() {
            return;
        }
        #[allow(clippy::cast_precision_loss)]
        let viewport = [size.0.max(1) as f32, size.1.max(1) as f32];
        // A perspective's second row is its 1 / tan(half the vertical field of view), turned by the view's rotation.
        let tan_half_fov_y = 1.0 / Vec3::new(view_proj.x_axis.y, view_proj.y_axis.y, view_proj.z_axis.y).length().max(1e-4);
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                view_proj: view_proj.to_cols_array_2d(),
                camera_pos: camera_pos.to_array(),
                pixel_angle: 2.0 * tan_half_fov_y.max(1e-4) / viewport[1],
                viewport,
                _pad: [0.0; 2],
            }),
        );
        queue.write_buffer(&self.corners, 0, bytemuck::cast_slice(&corners));
        #[allow(clippy::cast_possible_truncation)]
        {
            self.count = corners.len() as u32;
        }
    }

    /// Draw the prepared channels into the HDR scene.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, hdr: &wgpu::TextureView) {
        let (Some(bind_group), true) = (self.bind_group.as_ref(), self.count > 0) else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Lightning"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, self.corners.slice(..));
        pass.draw(0..self.count, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strike() -> Bolt {
        Bolt { from: Vec3::new(0.0, 1200.0, 0.0), to: Vec3::new(150.0, 0.0, -80.0), seed: 7, candela_per_metre: Vec3::splat(2.0e7), branching: 0.7, grown: 1.0 }
    }

    /// The main channel is one unbroken line from its start to its end, crooked but not wandering off.
    #[test]
    fn a_channel_joins_its_ends_and_stays_near_the_line() {
        let bolt = strike();
        let mut spans = Vec::new();
        bolt.grow(&mut spans);
        let main: Vec<&Span> = spans.iter().filter(|s| s.level == 1.0).collect();
        assert!(main.len() > 40, "{} spans", main.len());
        assert!(main[0].a.distance(bolt.from) < 1e-3 && main.last().unwrap().b.distance(bolt.to) < 1e-3);
        for pair in main.windows(2) {
            assert!(pair[0].b.distance(pair[1].a) < 1e-3, "a break in the channel");
        }
        let line = (bolt.to - bolt.from).normalize();
        for s in &main {
            let off = (s.a - bolt.from).cross(line).length();
            assert!(off < 0.35 * bolt.from.distance(bolt.to), "{off} m off the line");
        }
        // Crooked: its path is longer than the straight line, as a real channel's is.
        let path: f32 = main.iter().map(|s| s.a.distance(s.b)).sum();
        assert!(path > 1.03 * bolt.from.distance(bolt.to) && path < 1.6 * bolt.from.distance(bolt.to), "{path}");
    }

    /// A bolt whose end is not a place grows nothing (and returns: it used to recurse until the stack ran out).
    #[test]
    fn a_bolt_with_no_place_grows_nothing() {
        let mut spans = Vec::new();
        Bolt { to: Vec3::new(f32::NAN, 0.0, 0.0), ..strike() }.grow(&mut spans);
        Bolt { from: Vec3::new(0.0, f32::INFINITY, 0.0), ..strike() }.grow(&mut spans);
        assert!(spans.is_empty());
    }

    #[test]
    fn forks_are_dimmer_and_the_same_seed_grows_the_same_bolt() {
        let bolt = strike();
        let (mut one, mut two) = (Vec::new(), Vec::new());
        bolt.grow(&mut one);
        bolt.grow(&mut two);
        assert_eq!(one, two);
        assert!(one.iter().any(|s| s.level < 1.0), "a branching bolt forks");
        assert!(one.iter().all(|s| s.level <= 1.0 && s.level > 0.0));
        let mut plain = Vec::new();
        Bolt { branching: 0.0, ..bolt }.grow(&mut plain);
        assert!(plain.iter().all(|s| s.level == 1.0));
        assert!(one.len() <= MAX_SPANS);
    }

    #[test]
    fn a_growing_bolt_is_its_first_part() {
        let mut spans = Vec::new();
        Bolt { grown: 0.5, branching: 0.0, ..strike() }.grow(&mut spans);
        let mut whole = Vec::new();
        Bolt { branching: 0.0, ..strike() }.grow(&mut whole);
        assert!(spans.len() < whole.len() && spans.len() > whole.len() / 3);
        assert_eq!(spans[..], whole[..spans.len()]);
    }
}
