//! The photographic tone curve, run on the GPU from the very source the picture uses (`tone_curves.wgsl`), held to
//! the camera response it was fitted to (sunscar-photoreal, 6 Nov 2026; rimeborn-fp tools/photoreal).
#[allow(dead_code)]
#[path = "support/gpu.rs"]
mod gpu;

/// A digital camera's JPEG response to a grey ramp: display-linear output at stops from middle grey (0.18).
const CAMERA: [(f32, f32); 13] = [
    (-7.0, 0.0012),
    (-6.0, 0.0023),
    (-5.0, 0.0043),
    (-4.0, 0.0090),
    (-3.0, 0.020),
    (-2.0, 0.043),
    (-1.0, 0.089),
    (0.0, 0.181),
    (1.0, 0.33),
    (2.0, 0.54),
    (3.0, 0.83),
    (3.5, 0.93),
    (4.0, 0.98),
];

/// Grey at these stops through `photographic(x, contrast)` and AgX, plus a saturated orange two stops over.
fn evaluate(stops: &[f32], contrast: f32) -> Vec<[f32; 4]> {
    let (device, queue) = gpu::device();
    let mut source = format!(
        "{}\n@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;\n@compute @workgroup_size(1) fn main() {{\n",
        include_str!("../src/shaders/tone_curves.wgsl")
    );
    for (i, s) in stops.iter().enumerate() {
        source += &format!(
            "out[{i}] = vec4<f32>(photographic(vec3<f32>(0.18 * exp2({s:.4})), {contrast:.4}), agx(vec3<f32>(0.18 * exp2({s:.4}))).g);\n"
        );
    }
    let n = stops.len();
    // An orange (0.35, 0.12, 0.03) at middle-grey luminance, two stops over: the hue test.
    source += &format!("out[{n}] = vec4<f32>(photographic(vec3<f32>(0.35, 0.12, 0.03) * (0.18 / 0.1624) * 4.0, 1.0), 0.0);\n");
    source += "}";
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Tone curves"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let size = ((n + 1) * 16) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: output.as_entire_binding() }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &binding, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let (send, recv) = std::sync::mpsc::channel();
    readback.slice(..).map_async(wgpu::MapMode::Read, move |r| send.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    recv.recv().unwrap().unwrap();
    let data = readback.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice::<u8, [f32; 4]>(&data).to_vec()
}

#[test]
#[ignore = "Requires a graphics adapter; run for tone-curve changes"]
fn the_photographic_curve_holds_a_cameras_response_and_never_crushes() {
    let mut stops: Vec<f32> = CAMERA.iter().map(|c| c.0).collect();
    stops.extend([-10.0, -8.0, 5.0, 7.0]);
    let out = evaluate(&stops, 1.0);
    for (i, &(s, camera)) in CAMERA.iter().enumerate() {
        let [r, g, b, _] = out[i];
        assert!((r - g).abs() < 1e-5 && (g - b).abs() < 1e-5, "grey stays grey at {s} stops: {:?}", out[i]);
        let error = (g / camera).log2();
        let allowed = if s <= 2.0 { 0.12 } else { 0.4 };
        assert!(error.abs() <= allowed, "{s:+} stops: {g:.5} against the camera's {camera} ({error:+.2} stops)");
    }
    // Dark is not black: ten stops under middle grey is still about 1/255 on screen.
    let n = CAMERA.len();
    assert!(out[n][1] > 0.0002, "-10 stops: {}", out[n][1]);
    assert!(out[n + 1][1] > 0.0005, "-8 stops: {}", out[n + 1][1]);
    // A soft shoulder: +5 stops below white, +7 nearly there, and rising all the way.
    assert!(out[n + 2][1] > 0.8 && out[n + 2][1] < 0.95, "+5 stops: {}", out[n + 2][1]);
    assert!(out[n + 3][1] > out[n + 2][1] && out[n + 3][1] < 1.0, "+7 stops: {}", out[n + 3][1]);
    // The whole ramp rises.
    let mut ramp: Vec<(f32, f32)> = stops.iter().copied().zip(out.iter().map(|o| o[1])).collect();
    ramp.sort_by(|a, b| a.0.total_cmp(&b.0));
    for pair in ramp.windows(2) {
        assert!(pair[1].1 > pair[0].1, "falls between {} and {} stops", pair[0].0, pair[1].0);
    }
    // In the darks it keeps more than AgX does (AgX is 2 stops under the camera at -6), and agrees with AgX within
    // half a stop at middle grey.
    let six_under = CAMERA.iter().position(|c| c.0 == -6.0).unwrap();
    assert!(out[six_under][1] > out[six_under][3] * 2.5, "-6 stops: {} against AgX {}", out[six_under][1], out[six_under][3]);
    let grey = CAMERA.iter().position(|c| c.0 == 0.0).unwrap();
    assert!((out[grey][1] / out[grey][3]).log2().abs() < 0.5);
    // An orange two stops over keeps its order of channels (no skew past yellow) and stays coloured.
    let [r, g, b, _] = out[n + 4];
    assert!(r > g && g > b && r - b > 0.15, "orange at +2 stops: {r} {g} {b}");
}

#[test]
#[ignore = "Requires a graphics adapter; run for tone-curve changes"]
fn contrast_steepens_the_photographic_curve_without_crushing_it() {
    let stops = [-8.0, -4.0, 0.0, 2.0];
    let flat = evaluate(&stops, 1.0);
    let steep = evaluate(&stops, 1.3);
    assert!(steep[1][1] < flat[1][1], "more contrast darkens -4 stops");
    assert!(steep[3][1] > flat[3][1], "and brightens +2");
    assert!((steep[2][1] / flat[2][1] - 1.0).abs() < 0.02, "middle grey holds");
    assert!(steep[0][1] > 0.0001, "8 stops under is still above black: {}", steep[0][1]);
}
