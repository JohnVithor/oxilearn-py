#[derive(Debug, Clone)]
pub enum SpaceInfo {
    Discrete(usize),
    Continuous(Vec<(f32, f32)>),
}

impl SpaceInfo {
    pub fn is_discrete(&self) -> bool {
        match self {
            SpaceInfo::Discrete(_) => true,
            SpaceInfo::Continuous(_) => false,
        }
    }

    pub fn shape(&self) -> i64 {
        match self {
            SpaceInfo::Discrete(n) => *n as i64,
            SpaceInfo::Continuous(v) => v.len() as i64,
        }
    }
}
