//! A map of the local snowline (sunscar-photoreal, 7 Nov 2026): metres above sea level per cell, which snow.wgsl
//! reads where it covers (the High Eld's shade-driven snowline from sunscar-biomes); the look's own snowline holds
//! everywhere else.
//!
//! The file: `SNOWLN01`, then little-endian u32 width, u32 height, f32 x0, f32 z0 (scene metres of cell (0, 0)'s
//! corner), f32 cell size in metres, f32 reserved, then width x height f32 snowlines, row by row along +Z.

/// A parsed snowline map.
#[derive(Debug, Clone, PartialEq)]
pub struct SnowMap {
    pub width: u32,
    pub height: u32,
    pub x0: f32,
    pub z0: f32,
    pub cell: f32,
    pub metres: Vec<f32>,
}

const MAGIC: &[u8; 8] = b"SNOWLN01";
const HEADER: usize = 32;

/// Read a snowline map, refusing a wrong magic, a size the data does not fill exactly, or a cell that is not positive.
pub fn parse(bytes: &[u8]) -> Result<SnowMap, String> {
    if bytes.len() < HEADER || &bytes[..8] != MAGIC {
        return Err("not a snowline map (SNOWLN01)".into());
    }
    let word = |at: usize| [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]];
    let (width, height) = (u32::from_le_bytes(word(8)), u32::from_le_bytes(word(12)));
    let (x0, z0, cell) = (f32::from_le_bytes(word(16)), f32::from_le_bytes(word(20)), f32::from_le_bytes(word(24)));
    if width == 0 || height == 0 || width > 8192 || height > 8192 {
        return Err(format!("size {width} x {height} out of range"));
    }
    if !(cell > 0.0 && cell.is_finite() && x0.is_finite() && z0.is_finite()) {
        return Err(format!("bad placement: x0 {x0}, z0 {z0}, cell {cell}"));
    }
    let count = width as usize * height as usize;
    if bytes.len() != HEADER + count * 4 {
        return Err(format!("{} bytes of data for {width} x {height}", bytes.len() - HEADER));
    }
    let metres = bytes[HEADER..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok(SnowMap { width, height, x0, z0, cell, metres })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(width: u32, height: u32, cell: f32, data: &[f32]) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        for v in [width, height] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in [-100.0f32, 250.0, cell, 0.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn a_snowline_map_reads_back() {
        let map = parse(&file(2, 3, 16.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        assert_eq!((map.width, map.height, map.x0, map.z0, map.cell), (2, 3, -100.0, 250.0, 16.0));
        assert_eq!(map.metres, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn a_wrong_or_short_file_is_refused() {
        assert!(parse(b"SNOWLN02").is_err());
        assert!(parse(&file(2, 2, 16.0, &[1.0, 2.0, 3.0])).is_err());
        assert!(parse(&file(2, 1, 0.0, &[1.0, 2.0])).is_err());
        assert!(parse(&file(0, 1, 16.0, &[])).is_err());
    }
}
