use rand::{rngs::SmallRng, Rng, SeedableRng};
use tch::{Device, IndexOp, Kind, Tensor};

pub struct RandomExperienceBuffer {
    obs_size: i64,
    curr_states: Tensor,
    curr_actions: Tensor,
    rewards: Tensor,
    next_states: Tensor,
    dones: Tensor,
    size: i64,
    next_idx: i64,
    capacity: i64,
    minsize: i64,
    rng: SmallRng,
    device: Device,
}

impl Default for RandomExperienceBuffer {
    fn default() -> Self {
        Self {
            obs_size: 1,
            curr_states: Tensor::new(),
            curr_actions: Tensor::new(),
            rewards: Tensor::new(),
            next_states: Tensor::new(),
            dones: Tensor::new(),
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
    pub fn new(capacity: i64, obs_size: i64, minsize: i64, seed: u64, device: Device) -> Self {
        Self {
            obs_size,
            curr_states: Tensor::empty([capacity, obs_size], (Kind::Float, device)),
            curr_actions: Tensor::empty(capacity, (Kind::Int64, device)),
            rewards: Tensor::empty(capacity, (Kind::Float, device)),
            next_states: Tensor::empty([capacity, obs_size], (Kind::Float, device)),
            dones: Tensor::empty(capacity, (Kind::Int8, device)),
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
        let index: i64 = self.obs_size * self.next_idx;
        let index = Vec::from_iter(index..(index + self.obs_size));
        let index = &Tensor::from_slice(&index).to_device(self.device);

        let curr_state = &curr_state.to_device(self.device);
        let curr_action = &Tensor::from(curr_action as i64).to_device(self.device);
        let reward = &Tensor::from(reward).to_device(self.device);
        let done = &Tensor::from(done as i8).to_device(self.device);
        let next_state = &next_state.to_device(self.device);

        self.curr_states = self.curr_states.put(index, curr_state, false);
        self.next_states = self.next_states.put(index, next_state, false);

        let index = &Tensor::from(self.next_idx).to_device(self.device);

        self.curr_actions = self.curr_actions.put(index, curr_action, false);
        self.rewards = self.rewards.put(index, reward, false);
        self.dones = self.dones.put(index, done, false);

        self.next_idx = (self.next_idx + 1) % self.capacity;
        self.size = self.capacity.min(self.size + 1);
    }

    pub fn sample_batch(&mut self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let index: Vec<i64> = (0..size)
            .map(|_| self.rng.gen_range(0..self.size))
            .collect();
        (
            self.curr_states.i(index.clone()).to_device(self.device),
            self.curr_actions
                .i(index.clone())
                .reshape([-1, 1])
                .to_device(self.device),
            self.rewards
                .i(index.clone())
                .reshape([-1, 1])
                .to_device(self.device),
            self.dones
                .i(index.clone())
                .reshape([-1, 1])
                .to_device(self.device),
            self.next_states.i(index).to_device(self.device),
        )
    }
}
