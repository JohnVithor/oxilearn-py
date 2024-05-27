use rand::distributions::Distribution;
use rand::{rngs::SmallRng, SeedableRng};
use tch::{Device, IndexOp, Kind, Scalar, Tensor};
use torch_sys::IntList;
pub struct ExperienceStats {
    means: Tensor,
    msqs: Tensor,
    count: Tensor,
}

impl ExperienceStats {
    pub fn new(shape: impl IntList + Copy, device: Device) -> Self {
        Self {
            means: Tensor::zeros(shape, (Kind::Float, device)),
            msqs: Tensor::ones(shape, (Kind::Float, device)),
            count: Tensor::zeros(1, (Kind::Int64, device)),
        }
    }

    pub fn push(&mut self, value: &Tensor) {
        self.count += 1;
        let delta = value - &self.means;
        self.means += &delta / &self.count;
        let delta2 = value - &self.means;
        self.msqs += delta * delta2;
    }

    pub fn mean(&self) -> &Tensor {
        &self.means
    }

    pub fn var(&self) -> Tensor {
        &self.msqs / Scalar::float((self.count.int64_value(&[0]) - 1) as f64)
    }
}

pub struct RandomExperienceBuffer {
    obs_size: i64,
    pub curr_states: Tensor,
    curr_actions: Tensor,
    rewards: Tensor,
    pub next_states: Tensor,
    dones: Tensor,
    size: i64,
    next_idx: i64,
    capacity: i64,
    minsize: i64,
    rng: SmallRng,
    device: Device,
    pub stats: ExperienceStats,
    normalize_obs: bool,
}

impl RandomExperienceBuffer {
    pub fn new(
        capacity: i64,
        obs_size: i64,
        minsize: i64,
        seed: u64,
        normalize_obs: bool,
        device: Device,
    ) -> Self {
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
            stats: ExperienceStats::new(obs_size, device),
            normalize_obs,
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

        if self.normalize_obs {
            self.stats.push(curr_state);
        }
    }

    pub fn normalize(&self, values: Tensor) -> Tensor {
        // let (var, mean) = self.curr_states.var_mean_dim(0, false, false);
        // ((values - mean) / var.sqrt()).to_device(self.device)
        if self.normalize_obs {
            (values - self.stats.mean()) / (self.stats.var()).sqrt()
        } else {
            values
        }
    }

    pub fn sample_batch(&mut self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let dist: rand::distributions::Uniform<i64> =
            rand::distributions::Uniform::new(0i64, self.size as i64);
        let index: Vec<i64> = dist.sample_iter(&mut self.rng).take(size).collect();
        // println!("{index:?}");
        (
            self.normalize(self.curr_states.i(index.clone()))
                .to_kind(Kind::Float),
            self.curr_actions.i(index.clone()).reshape([-1, 1]),
            self.rewards.i(index.clone()).reshape([-1, 1]),
            self.dones.i(index.clone()).reshape([-1, 1]),
            self.normalize(self.next_states.i(index))
                .to_kind(Kind::Float),
        )
    }
}
