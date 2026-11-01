use anyhow::{bail, ensure, Result};
use std::collections::{BTreeSet, HashSet};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Visibility,
    Hdr,
    History,
    Display,
    PathQueue,
    IndirectDispatch,
}
#[derive(Debug, Clone)]
pub struct Resource {
    pub name: String,
    pub format: Format,
    pub owner: usize,
    pub persistent: bool,
}
#[derive(Debug, Clone)]
pub struct Pass {
    pub name: String,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}
#[derive(Debug, Default)]
pub struct Graph {
    pub passes: Vec<Pass>,
    pub resources: Vec<Resource>,
}
#[derive(Debug)]
pub struct Plan {
    pub order: Vec<usize>,
    pub barriers: Vec<(usize, usize, usize)>,
}
impl Graph {
    pub fn prototype() -> Self {
        Self {
            passes: vec![
                Pass {
                    name: "visibility".into(),
                    inputs: vec![],
                    outputs: vec![0],
                },
                Pass {
                    name: "shading".into(),
                    inputs: vec![0],
                    outputs: vec![1, 5, 6, 7],
                },
                Pass {
                    name: "accumulation".into(),
                    inputs: vec![1],
                    outputs: vec![2, 3],
                },
                Pass {
                    name: "tonemap".into(),
                    inputs: vec![0, 3],
                    outputs: vec![4],
                },
            ],
            resources: [
                ("visibility", Format::Visibility, false, 0),
                ("radiance", Format::Hdr, false, 1),
                ("history", Format::History, true, 2),
                ("filtered", Format::Hdr, false, 2),
                ("display", Format::Display, false, 3),
                ("ray_queue_a", Format::PathQueue, false, 1),
                ("ray_queue_b", Format::PathQueue, false, 1),
                ("queue_control", Format::IndirectDispatch, false, 1),
            ]
            .into_iter()
            .map(|(name, format, persistent, owner)| Resource {
                name: name.into(),
                format,
                owner,
                persistent,
            })
            .collect(),
        }
    }
    pub fn compile(&self) -> Result<Plan> {
        let mut names = HashSet::new();
        for p in &self.passes {
            ensure!(names.insert(&p.name), "duplicate pass name");
        }
        let mut writers = vec![None; self.resources.len()];
        for (i, p) in self.passes.iter().enumerate() {
            for &r in &p.outputs {
                let resource = self
                    .resources
                    .get(r)
                    .ok_or_else(|| anyhow::anyhow!("invalid output"))?;
                ensure!(resource.owner == i, "output not owned by pass");
                ensure!(writers[r].replace(i).is_none(), "multiple writers");
            }
        }
        let mut edges = vec![BTreeSet::new(); self.passes.len()];
        let mut degree = vec![0; self.passes.len()];
        let mut barriers = Vec::new();
        for (i, p) in self.passes.iter().enumerate() {
            for &r in &p.inputs {
                let writer = writers
                    .get(r)
                    .copied()
                    .flatten()
                    .ok_or_else(|| anyhow::anyhow!("input has no producer"))?;
                ensure!(
                    writer != i,
                    "same-frame feedback forbidden; history is pass-local"
                );
                if edges[writer].insert(i) {
                    degree[i] += 1;
                }
                barriers.push((writer, i, r));
            }
        }
        let mut ready: BTreeSet<_> = degree
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (d == 0).then_some(i))
            .collect();
        let mut order = Vec::new();
        while let Some(i) = ready.pop_first() {
            order.push(i);
            for &j in &edges[i] {
                degree[j] -= 1;
                if degree[j] == 0 {
                    ready.insert(j);
                }
            }
        }
        if order.len() != self.passes.len() {
            bail!("render graph contains a cycle");
        }
        Ok(Plan { order, barriers })
    }
}
