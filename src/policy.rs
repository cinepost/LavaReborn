use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Interactive,
    Final,
}
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    pub batch_samples: u32,
    pub fast_build: bool,
    pub filter_radius: u32,
}
impl Mode {
    pub fn policy(self) -> Policy {
        match self {
            Self::Interactive => Policy {
                batch_samples: 1,
                fast_build: true,
                filter_radius: 1,
            },
            Self::Final => Policy {
                batch_samples: 16,
                fast_build: false,
                filter_radius: 1,
            },
        }
    }
}
/// Latest-wins scene mailbox. A render worker snapshots only at GPU batch boundaries.
/// Producers never wait for a GPU submission; replacing an Arc is constant-time.
pub struct Updates<T> {
    latest: std::sync::Mutex<Option<(u64, std::sync::Arc<T>)>>,
    revision: std::sync::atomic::AtomicU64,
}
impl<T> Default for Updates<T> {
    fn default() -> Self {
        Self {
            latest: Default::default(),
            revision: 0.into(),
        }
    }
}
impl<T> Updates<T> {
    pub fn publish(&self, value: T) -> u64 {
        let mut slot = self.latest.lock().unwrap();
        let revision = self
            .revision
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            + 1;
        *slot = Some((revision, std::sync::Arc::new(value)));
        revision
    }
    pub fn take(&self) -> Option<(u64, std::sync::Arc<T>)> {
        self.latest.lock().unwrap().take()
    }
    pub fn revision(&self) -> u64 {
        self.revision.load(std::sync::atomic::Ordering::Acquire)
    }
}
