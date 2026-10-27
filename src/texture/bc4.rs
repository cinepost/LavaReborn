//! Portable BC4 UNORM/SNORM encoder. Blocks are independent loader-pool jobs.
use super::CompressionQuality;
use rayon::prelude::*;
fn palette(a: i16, b: i16, signed: bool) -> [f32; 8] {
    let mut p = [0.; 8];
    p[0] = a as f32;
    p[1] = b as f32;
    if a > b {
        for i in 1..7 {
            p[i + 1] = ((7 - i) as f32 * p[0] + i as f32 * p[1]) / 7.;
        }
    } else {
        for i in 1..5 {
            p[i + 1] = ((5 - i) as f32 * p[0] + i as f32 * p[1]) / 5.;
        }
        p[6] = if signed { -127. } else { 0. };
        p[7] = if signed { 127. } else { 255. };
    }
    p
}
fn fit(values: &[f32; 16], a: i16, b: i16, signed: bool) -> (f32, u64) {
    let p = palette(a, b, signed);
    let mut error = 0.;
    let mut bits = 0;
    for (i, v) in values.iter().enumerate() {
        let mut best = 0;
        let mut distance = f32::INFINITY;
        for (j, q) in p.iter().enumerate() {
            let d = (v - q) * (v - q);
            if d < distance {
                best = j;
                distance = d;
            }
        }
        error += distance;
        bits |= (best as u64) << (3 * i);
    }
    (error, bits)
}
pub(super) fn encode(
    pixels: &[[f32; 4]],
    w: u32,
    h: u32,
    signed: bool,
    scale: f32,
    quality: CompressionQuality,
) -> Vec<u8> {
    let bw = w.div_ceil(4);
    let bh = h.div_ceil(4);
    let unit = if signed { 127. } else { 255. };
    let lower = if signed { -127 } else { 0 };
    let upper = if signed { 127 } else { 255 };
    let radius = match quality {
        CompressionQuality::Fast => 0,
        CompressionQuality::Balanced => 1,
        CompressionQuality::High => 2,
    };
    let blocks: Vec<[u8; 8]> = (0..bw * bh)
        .into_par_iter()
        .map(|block| {
            let (bx, by) = (block % bw, block / bw);
            let mut values = [0.; 16];
            for y in 0..4 {
                for x in 0..4 {
                    values[(y * 4 + x) as usize] = (pixels
                        [((by * 4 + y).min(h - 1) * w + (bx * 4 + x).min(w - 1)) as usize][0]
                        / scale
                        * unit)
                        .clamp(lower as f32, upper as f32);
                }
            }
            let a = values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .round() as i16;
            let b = values.iter().copied().fold(f32::INFINITY, f32::min).round() as i16;
            let mut best = (f32::INFINITY, 0, a, b);
            for da in -radius..=radius {
                for db in -radius..=radius {
                    let aa = (a + da).clamp(lower, upper);
                    let bb = (b + db).clamp(lower, upper);
                    let (error, bits) = fit(&values, aa, bb, signed);
                    if error < best.0 {
                        best = (error, bits, aa, bb);
                    }
                }
            }
            let mut out = [0; 8];
            out[0] = best.2 as u8;
            out[1] = best.3 as u8;
            out[2..].copy_from_slice(&best.1.to_le_bytes()[..6]);
            out
        })
        .collect();
    bytemuck::cast_slice(&blocks).to_vec()
}
#[cfg(test)]
pub(super) fn decode(block: &[u8], signed: bool) -> [f32; 16] {
    let endpoint = |v: u8| {
        if signed {
            (v as i8).max(-127) as i16
        } else {
            v as i16
        }
    };
    let p = palette(endpoint(block[0]), endpoint(block[1]), signed);
    let mut bits = [0; 8];
    bits[..6].copy_from_slice(&block[2..8]);
    let bits = u64::from_le_bytes(bits);
    std::array::from_fn(|i| p[((bits >> (3 * i)) & 7) as usize] / if signed { 127. } else { 255. })
}
