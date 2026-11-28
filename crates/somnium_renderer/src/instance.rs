//! Instance data management for the Visibility Buffer pipeline.

use bytemuck::{Pod, Zeroable};

/// `GpuInstanceData::_padding` of a draw that casts no screen-space contact shadow (`exclude_from_contact_shadows`).
/// Terrain chunks pack their LOD into the same word and always carry a vertex count in bits 4..13, so they never
/// equal it; the CDLOD morph reads its own bits, all zero here, and does nothing.
pub const NO_CONTACT_SHADOW: u32 = 0x8000_0000;

/// `GpuInstanceData::_padding` of an instance-batch copy that hands over to another LOD at a band edge (Sunscar,
/// 6 Nov 2026), from [`crate::instance_batch::InstanceBatch::fade`]: `edge` > 0 fades the copy out past that
/// horizontal distance (the near band), `edge` < 0 fades it in past `-edge` (the far band), 0 is no fade (0).
/// The edge in 1/16 m sits in bits 14..29 and the direction in bit 0 (set: in). Bits 4..13 stay clear, where every
/// terrain chunk carries its vertex count, and bit 31 too, so neither the morph nor the contact-shadow test reads it.
/// `visibility.wgsl` dithers alpha-tested parts across the edge and switches opaque ones at it; `shadow.wgsl` switches.
#[must_use]
pub fn foliage_fade(edge: f32) -> u32 {
    if !edge.is_finite() || edge.abs() < 1.0 / 16.0 {
        return 0;
    }
    let q = (edge.abs() * 16.0).round().clamp(1.0, 65535.0) as u32;
    (q << 14) | u32::from(edge < 0.0)
}

/// Per-instance data matching the GPU layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuInstanceData {
    pub model_matrix: [[f32; 4]; 4],
    pub material_id: u32,
    pub mesh_vertex_offset: u32,
    pub mesh_index_offset: u32,
    pub _padding: u32,
}

/// Manages per-instance data in a GPU storage buffer.
pub struct InstancePool {
    pub buffer: wgpu::Buffer,
    instances: Vec<GpuInstanceData>,
    /// What the GPU buffer holds now, so a frame uploads only the chunks that changed.
    uploaded: Vec<GpuInstanceData>,
    /// Chunks the last upload wrote.
    written: usize,
}

/// Instances compared and uploaded together (80 KiB).
const UPLOAD_CHUNK: usize = 1024;

/// Write `now` into `buffer` (from its start), only the `chunk`-element runs that differ from `uploaded` (what the
/// buffer holds), and remember what was written. Returns how many runs went. An empty `uploaded` sends everything.
pub(crate) fn upload_changed<T: Pod>(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    now: &[T],
    uploaded: &mut Vec<T>,
    chunk: usize,
) -> usize {
    let n = now.len();
    uploaded.truncate(n);
    let stride = std::mem::size_of::<T>();
    let mut written = 0;
    for start in (0..n).step_by(chunk.max(1)) {
        let end = (start + chunk).min(n);
        let part = &now[start..end];
        let same = uploaded
            .get(start..end)
            .is_some_and(|was| bytemuck::cast_slice::<_, u8>(was) == bytemuck::cast_slice::<_, u8>(part));
        if same {
            continue;
        }
        queue.write_buffer(buffer, (start * stride) as u64, bytemuck::cast_slice(part));
        written += 1;
        if uploaded.len() < end {
            uploaded.resize(end, T::zeroed());
        }
        uploaded[start..end].copy_from_slice(part);
    }
    written
}

impl InstancePool {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global Instance Buffer"),
            size: 1024 * 1024 * 16, // 16MB
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            instances: Vec::with_capacity(256),
            uploaded: Vec::new(),
            written: 0,
        }
    }

    /// Clear instances for the current frame.
    pub fn clear(&mut self) {
        self.instances.clear();
    }

    /// Add an instance to the current frame's batch.
    pub fn add_instance(&mut self, data: GpuInstanceData) -> u32 {
        let id = self.instances.len() as u32;
        self.instances.push(data);
        id
    }

    /// How many 1,024-instance chunks the last upload wrote (the rest were already on the GPU).
    #[must_use]
    pub fn chunks_written(&self) -> usize {
        self.written
    }

    /// This frame's instances, in buffer order.
    #[must_use]
    pub fn instances(&self) -> &[GpuInstanceData] {
        &self.instances
    }

    /// Make room for this frame's instances. A region holds more than the first 16 MiB fits (about 209,000): the
    /// buffer doubles, up to the device's storage binding limit, past which the last instances are dropped with a
    /// warning rather than written out of bounds. Returns whether there is a new buffer, which the bind groups that
    /// name it must be rebuilt for.
    pub fn reserve(&mut self, device: &wgpu::Device) -> bool {
        self.reserve_extent(device, 0)
    }

    /// [`Self::reserve`], with room for `extent` instances at least: instance batches (`crate::instance_batch`)
    /// keep theirs past the frame's own. The GPU rewrites those when this returns true.
    pub fn reserve_extent(&mut self, device: &wgpu::Device, extent: usize) -> bool {
        let stride = std::mem::size_of::<GpuInstanceData>() as u64;
        let need = self.instances.len().max(extent) as u64 * stride;
        if need <= self.buffer.size() {
            return false;
        }
        let limit = u64::from(device.limits().max_storage_buffer_binding_size) / stride * stride;
        let size = need.next_power_of_two().min(limit).max(self.buffer.size());
        if size < need {
            let keep = ((size / stride) as usize).min(self.instances.len());
            tracing::warn!(
                instances = self.instances.len(),
                keep,
                "more instances than the device can bind; the rest are not drawn this frame"
            );
            self.instances.truncate(keep);
        }
        if size == self.buffer.size() {
            return false;
        }
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global Instance Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.uploaded.clear();
        true
    }

    /// Upload this frame's instances: only the chunks that differ from what the buffer already holds. A still
    /// region with a few moving people sends the few chunks they are in, not every house again.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        self.written = upload_changed(queue, &self.buffer, &self.instances, &mut self.uploaded, UPLOAD_CHUNK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fade_word_is_neither_terrain_nor_the_contact_flag() {
        for edge in [45.0f32, -45.0, 0.07, -4000.0, 4095.9] {
            let w = foliage_fade(edge);
            assert_ne!(w, 0);
            assert_ne!(w, NO_CONTACT_SHADOW);
            assert_eq!((w >> 4) & 1023, 0, "terrain's vertex count and morph bit stay clear");
            assert_eq!(w & 1, u32::from(edge < 0.0));
            assert!((f32::from((w >> 14) as u16) / 16.0 - edge.abs()).abs() <= 1.0 / 32.0 + 1e-3);
        }
        assert_eq!(foliage_fade(0.0), 0);
        assert_eq!(foliage_fade(f32::NAN), 0);
    }
}
