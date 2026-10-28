//! Asynchronous UDIM directory discovery; each concrete tile uses LoadRequest.
use super::*;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdimTile {
    pub number: u32,
    pub file: PathBuf,
}
pub fn is_udim(path: &Path) -> bool {
    {
        let path = path.to_string_lossy();
        path.contains("<UDIM>") || path.contains("%(UDIM)d")
    }
}
/// Standard four-digit UDIMs: 1001 + u + 10*v, with 0 <= u < 10.
pub fn tile_number(u: i32, v: i32) -> Option<u32> {
    if !(0..10).contains(&u) || !(0..900).contains(&v) {
        return None;
    }
    let n = 1001 + u as u32 + 10 * v as u32;
    (n <= 9999).then_some(n)
}
pub fn discover(pattern: &Path) -> Result<Vec<UdimTile>> {
    let normalized = super::normalize_path(pattern);
    let pattern = normalized.as_path();
    let name = pattern
        .file_name()
        .and_then(|n| n.to_str())
        .context("UDIM filename must be UTF-8")?;
    ensure!(
        name.matches("<UDIM>").count() == 1,
        "UDIM requires exactly one <UDIM> token in the filename"
    );
    let (prefix, suffix) = name.split_once("<UDIM>").unwrap();
    let parent = pattern
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    ensure!(
        !is_udim(parent),
        "UDIM tokens in directories are unsupported"
    );
    let mut tiles = Vec::new();
    for entry in
        std::fs::read_dir(parent).with_context(|| format!("UDIM directory {}", parent.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(number) = name
            .strip_prefix(prefix)
            .and_then(|s| s.strip_suffix(suffix))
        else {
            continue;
        };
        if number.len() != 4 || !number.bytes().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let number: u32 = number.parse()?;
        if !(1001..=9999).contains(&number) || !entry.path().is_file() {
            continue;
        }
        tiles.push(UdimTile {
            number,
            file: std::fs::canonicalize(entry.path())?,
        });
    }
    tiles.sort_by_key(|t| t.number);
    Ok(tiles)
}
struct State {
    tiles: Vec<UdimTile>,
    scanned: bool,
    pending: bool,
    generation: u64,
    error: Option<String>,
}
struct Done {
    request: LoadRequest,
    generation: u64,
    result: Result<Vec<UdimTile>>,
}
pub(crate) struct Catalog {
    states: HashMap<LoadRequest, State>,
    sender: mpsc::Sender<Done>,
    receiver: mpsc::Receiver<Done>,
    scan: Instant,
    serial: u64,
}
impl Catalog {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            states: HashMap::new(),
            sender,
            receiver,
            scan: Instant::now(),
            serial: 0,
        }
    }
    pub fn configure(&mut self, patterns: &std::collections::HashSet<LoadRequest>) {
        self.states.retain(|r, _| patterns.contains(r));
        for request in patterns {
            self.states.entry(request.clone()).or_insert(State {
                tiles: Vec::new(),
                scanned: false,
                pending: false,
                generation: 0,
                error: None,
            });
            if !self.states[request].pending {
                self.schedule(request.clone());
            }
        }
    }
    fn schedule(&mut self, request: LoadRequest) {
        self.serial += 1;
        let generation = self.serial;
        let state = self.states.get_mut(&request).unwrap();
        state.pending = true;
        state.generation = generation;
        let sender = self.sender.clone();
        pool().spawn(move || {
            let result = std::panic::catch_unwind(|| discover(&request.file))
                .unwrap_or_else(|_| Err(anyhow::anyhow!("UDIM discovery panicked")));
            let _ = sender.send(Done {
                request,
                generation,
                result,
            });
        });
    }
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(done) = self.receiver.try_recv() {
            let Some(state) = self.states.get_mut(&done.request) else {
                continue;
            };
            if state.generation != done.generation {
                continue;
            }
            state.pending = false;
            state.scanned = true;
            match done.result {
                Ok(tiles) => {
                    changed |= state.tiles != tiles;
                    state.tiles = tiles;
                    let error = state
                        .tiles
                        .is_empty()
                        .then(|| format!("no UDIM tiles match {}", done.request.file.display()));
                    if error != state.error {
                        if let Some(e) = &error {
                            println!("Lava [WARNING]: {e}; using material constant.");
                        }
                    }
                    state.error = error;
                }
                Err(e) => {
                    let message = format!("UDIM discovery {}: {e:#}", done.request.file.display());
                    if state.error.as_ref() != Some(&message) {
                        println!("Lava [WARNING]: {message}; retaining previous tiles.");
                    }
                    state.error = Some(message);
                }
            }
        }
        if self.scan.elapsed() >= Duration::from_millis(250) {
            self.scan = Instant::now();
            let requests: Vec<_> = self
                .states
                .iter()
                .filter(|(_, s)| !s.pending)
                .map(|(r, _)| r.clone())
                .collect();
            for request in requests {
                self.schedule(request);
            }
        }
        changed
    }
    pub fn tiles(&self, request: &LoadRequest) -> &[UdimTile] {
        self.states
            .get(request)
            .map(|s| s.tiles.as_slice())
            .unwrap_or(&[])
    }
    pub fn pending(&self) -> bool {
        self.states.values().any(|s| !s.scanned)
    }
    pub fn scanning(&self) -> bool {
        self.states.values().any(|s| s.pending)
    }
    pub fn check_errors(&self) -> Result<()> {
        for state in self.states.values() {
            if let Some(e) = &state.error {
                bail!("final texture preparation failed: {e}");
            }
        }
        Ok(())
    }
}
