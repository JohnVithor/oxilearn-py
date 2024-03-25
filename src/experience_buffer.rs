use numpy::ndarray::Array1;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tch::{Device, Tensor};

pub struct RandomExperienceBuffer {
    curr_states: Array1<Tensor>,
    curr_actions: Array1<Tensor>,
    rewards: Array1<Tensor>,
    next_states: Array1<Tensor>,
    dones: Array1<Tensor>,
    size: usize,
    next_idx: usize,
    capacity: usize,
    minsize: usize,
    rng: SmallRng,
    device: Device,
}

impl Default for RandomExperienceBuffer {
    fn default() -> Self {
        Self {
            curr_states: Array1::default(10_000),
            curr_actions: Array1::default(10_000),
            rewards: Array1::default(10_000),
            next_states: Array1::default(10_000),
            dones: Array1::default(10_000),
            size: 0,
            next_idx: 0,
            capacity: 10_000,
            minsize: 1_000,
            rng: SmallRng::seed_from_u64(42),
            device: Device::cuda_if_available(),
        }
    }
}

impl RandomExperienceBuffer {
    pub fn new(capacity: usize, minsize: usize, seed: u64, device: Device) -> Self {
        Self {
            curr_states: Array1::default(capacity),
            curr_actions: Array1::default(capacity),
            rewards: Array1::default(capacity),
            next_states: Array1::default(capacity),
            dones: Array1::default(capacity),
            capacity,
            next_idx: 0,
            size: 0,
            minsize,
            rng: SmallRng::seed_from_u64(seed),
            device,
        }
    }

    pub fn ready(&self) -> bool {
        self.size >= self.minsize
    }

    pub fn add(
        &mut self,
        curr_state: &Tensor,
        curr_action: usize,
        reward: f32,
        done: bool,
        next_state: &Tensor,
    ) {
        self.curr_states[self.next_idx] = curr_state.shallow_clone();
        self.curr_actions[self.next_idx] = Tensor::from(curr_action as i64);
        self.rewards[self.next_idx] = Tensor::from(reward);
        self.dones[self.next_idx] = Tensor::from(done as i64);
        self.next_states[self.next_idx] = next_state.shallow_clone();

        self.next_idx = (self.next_idx + 1) % self.capacity;
        self.size = self.capacity.min(self.size + 1);
    }

    pub fn sample_batch(&mut self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let index: Vec<usize> = (0..size)
            .map(|_| self.rng.gen_range(0..self.size))
            .collect();
        let mut curr_obs: Vec<Tensor> = Vec::new();
        let mut curr_actions: Vec<Tensor> = Vec::new();
        let mut rewards: Vec<Tensor> = Vec::new();
        let mut dones: Vec<Tensor> = Vec::new();
        let mut next_obs: Vec<Tensor> = Vec::new();
        index.iter().for_each(|i| {
            curr_obs.push(self.curr_states[*i].shallow_clone());
            curr_actions.push(self.curr_actions[*i].shallow_clone());
            rewards.push(self.rewards[*i].shallow_clone());
            dones.push(self.dones[*i].shallow_clone());
            next_obs.push(self.next_states[*i].shallow_clone());
        });
        (
            Tensor::stack(&curr_obs, 0)
                .to_kind(tch::Kind::Float)
                .to_device(self.device),
            Tensor::stack(&curr_actions, 0)
                .reshape([-1, 1])
                .to_device(self.device),
            Tensor::stack(&rewards, 0)
                .reshape([-1, 1])
                .to_kind(tch::Kind::Float)
                .to_device(self.device),
            Tensor::stack(&dones, 0)
                .reshape([-1, 1])
                .to_kind(tch::Kind::Float)
                .to_device(self.device),
            Tensor::stack(&next_obs, 0)
                .to_kind(tch::Kind::Float)
                .to_device(self.device),
        )
    }
}
