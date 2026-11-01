//! Shared light sampling and emitter-intersection hierarchy, independent of passes.
//! Selection PDFs are evaluated through the same parent chain used for sampling.
use crate::{
    scene::{GpuLight, Light},
    vulkan::{Buffer, Device},
};
use anyhow::{ensure, Result};
use ash::vk;
use glam::Vec3;
use std::sync::Arc;
pub const LEAF: u32 = u32::MAX;
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightNode {
    pub min_power: [f32; 4],
    pub max_infinite: [f32; 4],
    pub links: [u32; 4], // left, right, light index (MAX for interior), parent
}
#[derive(Default)]
pub struct LightTree {
    pub nodes: Vec<LightNode>,
    pub leaves: Vec<u32>,
}
impl LightTree {
    pub fn build(lights: &[GpuLight]) -> Self {
        let mut tree = Self {
            nodes: Vec::new(),
            leaves: vec![0; lights.len()],
        };
        let mut indices: Vec<usize> = (0..lights.len()).collect();
        if !indices.is_empty() {
            tree.build_node(lights, &mut indices, LEAF);
        }
        tree
    }
    fn build_node(&mut self, lights: &[GpuLight], indices: &mut [usize], parent: u32) -> u32 {
        let index = self.nodes.len() as u32;
        self.nodes.push(LightNode {
            min_power: [0.; 4],
            max_infinite: [0.; 4],
            links: [0, 0, LEAF, parent],
        });
        if indices.len() == 1 {
            let id = indices[0];
            let light = lights[id];
            let p = Vec3::from_slice(&light.position_kind);
            let area = Vec3::from_slice(&light.u_soft)
                .cross(Vec3::from_slice(&light.v))
                .length();
            let extent = if light.position_kind[3] == 2. {
                (Vec3::from_slice(&light.u_soft).abs() + Vec3::from_slice(&light.v).abs()) * 0.5
            } else {
                Vec3::ZERO
            };
            let power = Vec3::from_slice(&light.color_power).dot(Vec3::new(0.2126, 0.7152, 0.0722))
                * light.color_power[3]
                * light.response[0].max(light.response[1])
                * if light.position_kind[3] == 2. {
                    area
                } else {
                    1.
                };
            self.nodes[index as usize] = LightNode {
                min_power: [
                    p.x - extent.x,
                    p.y - extent.y,
                    p.z - extent.z,
                    power.max(1e-12),
                ],
                max_infinite: [
                    p.x + extent.x,
                    p.y + extent.y,
                    p.z + extent.z,
                    if light.position_kind[3] == 3. { 1. } else { 0. },
                ],
                links: [0, 0, id as u32, parent],
            };
            self.leaves[id] = index;
            return index;
        }
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for &i in indices.iter() {
            let p = Vec3::from_slice(&lights[i].position_kind);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let axis = (hi - lo).max_element();
        let axis = if hi.x - lo.x == axis {
            0
        } else if hi.y - lo.y == axis {
            1
        } else {
            2
        };
        indices.sort_unstable_by(|&a, &b| {
            lights[a].position_kind[axis].total_cmp(&lights[b].position_kind[axis])
        });
        let mid = indices.len() / 2;
        let (a, b) = indices.split_at_mut(mid);
        let left = self.build_node(lights, a, index);
        let right = self.build_node(lights, b, index);
        let l = self.nodes[left as usize];
        let r = self.nodes[right as usize];
        let lo = Vec3::from_slice(&l.min_power).min(Vec3::from_slice(&r.min_power));
        let hi = Vec3::from_slice(&l.max_infinite).max(Vec3::from_slice(&r.max_infinite));
        self.nodes[index as usize] = LightNode {
            min_power: [lo.x, lo.y, lo.z, l.min_power[3] + r.min_power[3]],
            max_infinite: [hi.x, hi.y, hi.z, l.max_infinite[3].max(r.max_infinite[3])],
            links: [left, right, LEAF, parent],
        };
        index
    }
    pub fn importance(&self, node: u32, position: Vec3) -> f32 {
        let n = self.nodes[node as usize];
        if n.max_infinite[3] > 0. {
            return n.min_power[3];
        }
        let lo = Vec3::from_slice(&n.min_power);
        let hi = Vec3::from_slice(&n.max_infinite);
        let delta = (lo - position).max(Vec3::ZERO) + (position - hi).max(Vec3::ZERO);
        n.min_power[3] / (delta.length_squared() + (hi - lo).length_squared() * 0.25 + 1e-4)
    }
    pub fn pmf(&self, light: usize, position: Vec3) -> f32 {
        let mut node = self.leaves[light];
        let mut pdf = 1.;
        while self.nodes[node as usize].links[3] != LEAF {
            let parent = self.nodes[node as usize].links[3];
            let p = self.nodes[parent as usize];
            let a = self.importance(p.links[0], position);
            let b = self.importance(p.links[1], position);
            let probability = (a / (a + b)).clamp(1e-4, 1. - 1e-4);
            pdf *= if node == p.links[0] {
                probability
            } else {
                1. - probability
            };
            node = parent;
        }
        pdf
    }
}
/// Any pass can bind these two read-only resources using light_bvh.glsl.
pub struct LightBvh {
    pub lights: Buffer,
    pub nodes: Buffer,
    pub count: u32,
    pub source: Vec<Light>,
}
impl LightBvh {
    pub fn upload(device: Arc<Device>, scene: &crate::scene::Scene) -> Result<Arc<Self>> {
        let mut lights = scene.gpu_lights();
        let tree = LightTree::build(&lights);
        ensure!(
            tree.nodes
                .iter()
                .flat_map(|n| n.min_power.iter().chain(n.max_infinite.iter()))
                .all(|v| v.is_finite()),
            "light BVH bounds/power exceed finite float range"
        );
        let distant: Vec<usize> = lights
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                (l.position_kind[3] == 3. && l.response[2] > 1e-12 && l.response[3] > 0.)
                    .then_some(i)
            })
            .collect();
        let first = distant.first().map_or(LEAF, |i| *i as u32);
        for (i, l) in lights.iter_mut().enumerate() {
            l.bvh[0] = tree.leaves[i];
            l.bvh[2] = LEAF; // Next finite distant emitter; no scan of local lights on misses.
            l.bvh[3] = first;
        }
        for pair in distant.windows(2) {
            lights[pair[0]].bvh[2] = pair[1] as u32;
        }
        let upload = |bytes: &[u8]| -> Result<Buffer> {
            let buffer = Buffer::new(
                device.clone(),
                bytes.len() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                true,
            )?;
            buffer.write(bytes)?;
            Ok(buffer)
        };
        Ok(Arc::new(Self {
            lights: upload(bytemuck::cast_slice(&lights))?,
            nodes: upload(bytemuck::cast_slice(&tree.nodes))?,
            count: lights.len() as u32,
            source: scene.lights.clone(),
        }))
    }
}
