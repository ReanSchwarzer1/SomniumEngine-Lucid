//! The ground's height where a mesh's pixel can read it (Sunscar, 11 Nov 2026).
//!
//! A rock set on a slope meets the ground along a hard line: scanned stone on one side, the terrain's layers on the
//! other, and nothing between them. The shading pass blends a mesh into the terrain where they meet
//! (`shading.wgsl` `ground_stand`), and for that it must know how high the ground is under a pixel that is not
//! ground. Nothing held that for it: the heightfield lived in the chunks' vertices.
//!
//! This is the heightfield as a texture, one texel per heightfield sample, registered in the bindless pool like the
//! terrain's other baked maps (the shading pass's fragment stage already holds the device's 16 sampled textures, and
//! the pool is one binding). RGB hold the height in metres above the terrain's own base as 24-bit fixed point: half
//! a millimetre a step from [`HEIGHT_FLOOR`] up, which is finer than a vertex's f32 at this world's distances. An
//! RGBA8 texture, because the pool is declared filterable and a 32-bit float is not; the shader reads four texels
//! and interpolates them itself, since the sampler cannot interpolate a number split across channels.

/// The lowest height the texture holds, in metres: below any sea floor this engine's worlds have.
pub const HEIGHT_FLOOR: f32 = -2048.0;
/// Steps per metre. With 24 bits the range is 8192 m: [`HEIGHT_FLOOR`] to 6144 m.
pub const STEPS_PER_METRE: f32 = 2048.0;

/// A terrain's height texture and the view the bindless pool holds.
pub struct GroundGpu {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

/// One height in metres as a texel. `shading.wgsl` `ground_decode` is its inverse.
#[must_use]
pub fn encode(metres: f32) -> [u8; 4] {
    let steps = ((metres - HEIGHT_FLOOR) * STEPS_PER_METRE).round().clamp(0.0, 16_777_215.0) as u32;
    [(steps >> 16) as u8, (steps >> 8) as u8, steps as u8, 255]
}

fn pack(heights: &[f32], height_scale: f32) -> Vec<u8> {
    heights.iter().flat_map(|height| encode(height * height_scale)).collect()
}

/// The texture for a heightfield of `total_x` by `total_z` samples (`heights` times `height_scale` is metres).
pub fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    heights: &[f32],
    total_x: u32,
    total_z: u32,
    height_scale: f32,
) -> GroundGpu {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Terrain ground height"),
        size: wgpu::Extent3d { width: total_x.max(1), height: total_z.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let gpu = GroundGpu { texture, view };
    rewrite(queue, &gpu, heights, total_x, total_z, height_scale);
    gpu
}

/// Write the heightfield again in place (a sculpt, a streamed tile taking the slot), keeping the bindless index.
pub fn rewrite(queue: &wgpu::Queue, gpu: &GroundGpu, heights: &[f32], total_x: u32, total_z: u32, height_scale: f32) {
    if heights.len() != (total_x * total_z) as usize || heights.is_empty() {
        return;
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &gpu.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pack(heights, height_scale),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(total_x * 4), rows_per_image: Some(total_z) },
        wgpu::Extent3d { width: total_x, height: total_z, depth_or_array_layers: 1 },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `shading.wgsl` `ground_decode` does with a texel.
    fn decode(texel: [u8; 4]) -> f32 {
        (f32::from(texel[0]) * 65536.0 + f32::from(texel[1]) * 256.0 + f32::from(texel[2])) / STEPS_PER_METRE
            + HEIGHT_FLOOR
    }

    #[test]
    fn a_height_comes_back_within_a_millimetre() {
        for metres in [-2048.0, -55.0, 0.0, 0.0004, 28.42, 783.137, 6143.9] {
            let back = decode(encode(metres));
            assert!((back - metres).abs() < 0.001, "{metres} came back as {back}");
        }
        // Out of range clamps to the ends rather than wrapping round to the other one.
        assert!(decode(encode(-9000.0)) <= HEIGHT_FLOOR + 0.001);
        assert!(decode(encode(9000.0)) > 6143.0);
    }

    #[test]
    fn the_packed_texture_is_the_scaled_heightfield_row_by_row() {
        let packed = pack(&[0.0, 0.5, 1.0, 2.0], 100.0);
        assert_eq!(packed.len(), 16);
        let heights: Vec<f32> = packed.chunks(4).map(|t| decode([t[0], t[1], t[2], t[3]])).collect();
        for (got, want) in heights.iter().zip([0.0, 50.0, 100.0, 200.0]) {
            assert!((got - want).abs() < 0.001);
        }
    }
}
