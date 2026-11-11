//! The sky light under a moving sun (sunscar-sky, 11 Nov 2026). While a day clock runs, `IblPass` builds the sky into
//! a second cube a few faces a frame and crossfades the bound cube to it, so the light follows the sun in straight
//! steps too small to see; rebuilt whole on every change, as a still scene's is, it would cost 36 passes a frame, and
//! rebuilt whole every second it would jump. Run on the GPU from the pass itself: a dusk sun sinking 0.01 degrees a
//! frame, the bound cube read every frame overhead (its roughest mip, the ambient), at the horizon under the sun (its
//! sharpest, the background) and away from it.
#[allow(dead_code)]
#[path = "support/gpu.rs"]
mod gpu;

use somnium_renderer::pass::atmosphere::AtmospherePass;
use somnium_renderer::pass::ibl::IblPass;

const PROBES: usize = 3;

/// Reads the bound cube along fixed directions and mips.
struct Probe {
    pipeline: wgpu::ComputePipeline,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl Probe {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky light probe"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var cube: texture_cube<f32>;
                 @group(0) @binding(1) var samp: sampler;
                 @group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>, 3>;
                 @compute @workgroup_size(1) fn main() {
                     out[0] = textureSampleLevel(cube, samp, vec3<f32>(0.0, 1.0, 0.0), 5.0);
                     out[1] = textureSampleLevel(cube, samp, normalize(vec3<f32>(0.6, 0.04, 0.8)), 0.0);
                     out[2] = textureSampleLevel(cube, samp, normalize(vec3<f32>(-0.6, 0.3, -0.8)), 2.0);
                 }"
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let size = (PROBES * 16) as u64;
        let buffer = |usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false })
        };
        Self {
            pipeline,
            output: buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            readback: buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        }
    }

    /// Luminance at each probe now.
    fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue, ibl: &IblPass) -> [f32; PROBES] {
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ibl.cube_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&ibl.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.output.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, (PROBES * 16) as u64);
        queue.submit([encoder.finish()]);
        let (send, recv) = std::sync::mpsc::channel();
        self.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| send.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        recv.recv().unwrap().unwrap();
        let texels = {
            let data = self.readback.slice(..).get_mapped_range().unwrap();
            bytemuck::cast_slice::<u8, [f32; 4]>(&data).to_vec()
        };
        self.readback.unmap();
        std::array::from_fn(|i| 0.2126 * texels[i][0] + 0.7152 * texels[i][1] + 0.0722 * texels[i][2])
    }
}

/// Toward a sun at `elevation` degrees, bearing fixed.
fn sun(elevation: f32) -> glam::Vec3 {
    let e = elevation.to_radians();
    glam::Vec3::new(0.6 * e.cos(), e.sin(), 0.8 * e.cos())
}

/// Its light on the ground, dimming as it sinks (as the transmittance would).
fn light(elevation: f32) -> glam::Vec3 {
    glam::Vec3::new(1.0, 0.92, 0.8) * 80_000.0 * (elevation / 6.0).clamp(0.05, 1.0)
}

#[test]
#[ignore = "Requires a graphics adapter; run for changes to the sky light's rebuild"]
fn a_running_clock_moves_the_sky_light_without_a_step() {
    let (device, queue) = gpu::device();
    let shaders = somnium_renderer::shaders::Shaders::new();
    let mut atmosphere = AtmospherePass::new(&device, &shaders);
    atmosphere.ensure_built(&device, &queue);
    let mut ibl = IblPass::new(&device, &shaders, &atmosphere);
    let probe = Probe::new(&device);

    // The first frame builds the cube whole, as a scene's first frame does.
    assert!(ibl.generate_if_needed(&device, &queue, sun(6.0), light(6.0), 0.0, None, false, None));
    let mut seen = vec![probe.read(&device, &queue, &ibl)];
    let frames = 300;
    for frame in 1..=frames {
        let elevation = 6.0 - 0.01 * frame as f32;
        ibl.generate_if_needed(&device, &queue, sun(elevation), light(elevation), 0.0, None, true, None);
        seen.push(probe.read(&device, &queue, &ibl));
    }
    for p in 0..PROBES {
        let lum: Vec<f32> = seen.iter().map(|s| s[p]).collect();
        let step = lum.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0_f32, f32::max);
        // The most the light moves across one cycle's length (18 frames building, 30 blending).
        let cycle = lum.windows(49).map(|w| (w[48] - w[0]).abs()).fold(0.0_f32, f32::max);
        let whole = (lum[frames] - lum[0]).abs();
        assert!(whole > 0.02 * lum[0].abs(), "probe {p}: the light must follow the sun ({} -> {})", lum[0], lum[frames]);
        // A crossfade over 30 frames moves a thirtieth of a cycle's change a frame; a rebuild whole moves all of it.
        assert!(step <= 0.1 * cycle, "probe {p}: a jump of {step} in one frame, against {cycle} over a cycle");
    }

    // A sun that jumps (a key that moves the clock an hour) is rebuilt at once, crossfade or not.
    let before = probe.read(&device, &queue, &ibl);
    assert!(ibl.generate_if_needed(&device, &queue, sun(20.0), light(20.0), 0.0, None, true, None));
    let after = probe.read(&device, &queue, &ibl);
    assert!(after[0] > 1.5 * before[0], "an hour on is there the same frame: {before:?} -> {after:?}");
}

/// Dusk (sunscar-sky, item 2): a sky lit by the sun above the air (`generate_if_needed`'s `ground_sun`) marches its own
/// transmittance and the planet's shadow, so it keeps its light after sunset and fades through the twilight, as the
/// real sky does: civil dusk's zenith a few cd/m2, nautical's a hundredth of that. Lit by the light on the ground, as
/// every other scene's sky is, it goes out with the sun's last ray.
#[test]
#[ignore = "Requires a graphics adapter; run for changes to the sky's twilight"]
fn a_sky_lit_above_the_air_keeps_its_light_after_sunset() {
    let (device, queue) = gpu::device();
    let shaders = somnium_renderer::shaders::Shaders::new();
    let mut atmosphere = AtmospherePass::new(&device, &shaders);
    atmosphere.ensure_built(&device, &queue);
    let mut ibl = IblPass::new(&device, &shaders, &atmosphere);
    let probe = Probe::new(&device);
    let above = glam::Vec3::splat(80_000.0);
    let mut overhead = |elevation: f32, lit_above: bool| {
        // Under the horizon the light on the ground is nothing (sun::transmittance), whichever way the sky is lit.
        let ground = if elevation > 0.0 { light(elevation) } else { glam::Vec3::ZERO };
        let sky_sun = if lit_above { above } else { ground };
        ibl.generate_if_needed(&device, &queue, sun(elevation), sky_sun, 0.0, None, false, lit_above.then_some(ground));
        probe.read(&device, &queue, &ibl)[0]
    };
    let day = overhead(35.0, true);
    let civil = overhead(-4.0, true);
    let nautical = overhead(-10.0, true);
    let night = overhead(-25.0, true);
    let gone = overhead(-4.0, false);
    println!("overhead sky light: day {day:.1}, -4 deg {civil:.3}, -10 deg {nautical:.5}, -25 deg {night:.7}; on the ground's light at -4: {gone:.7}");
    assert!(civil > 0.3 && civil < 60.0, "civil dusk's sky: {civil} cd/m2");
    assert!(nautical < civil / 20.0 && nautical > civil / 5000.0, "nautical dusk's: {nautical} against {civil}");
    assert!(night < nautical / 10.0, "night: {night}");
    assert!(gone < 0.01, "lit by the ground's light the sky is gone at -4 degrees: {gone}");
    assert!(day > 100.0 * civil, "day {day} against dusk {civil}");
}
