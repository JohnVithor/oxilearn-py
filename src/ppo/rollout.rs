use tch::{Device, Tensor};

pub struct Rollout {
    obs: Tensor,
    actions: Tensor,
    logprobs: Tensor,
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    episode_returns: Vec<f64>,
    step: usize,
}

impl Rollout {
    pub fn new(
        num_steps: usize,
        num_envs: usize,
        obs_shape: &[i64],
        action_shape: &[i64],
        device: Device,
    ) -> Self {
        Rollout {
            obs: Tensor::zeros(
                [num_steps as i64, num_envs as i64]
                    .iter()
                    .chain(obs_shape)
                    .cloned()
                    .collect::<Vec<_>>(),
                (tch::Kind::Float, device),
            ),
            actions: Tensor::zeros(
                [num_steps as i64, num_envs as i64]
                    .iter()
                    .chain(action_shape)
                    .cloned()
                    .collect::<Vec<_>>(),
                (tch::Kind::Float, device),
            ),
            logprobs: Tensor::zeros(
                [num_steps as i64, num_envs as i64],
                (tch::Kind::Float, device),
            ),
            rewards: Tensor::zeros(
                [num_steps as i64, num_envs as i64],
                (tch::Kind::Float, device),
            ),
            dones: Tensor::zeros(
                [num_steps as i64, num_envs as i64],
                (tch::Kind::Float, device),
            ),
            values: Tensor::zeros(
                [num_steps as i64, num_envs as i64],
                (tch::Kind::Float, device),
            ),
            episode_returns: Vec::new(),
            step: 0,
        }
    }

    pub fn add(
        &mut self,
        obs: Tensor,
        action: Tensor,
        logprob: Tensor,
        reward: Tensor,
        done: Tensor,
        value: Tensor,
    ) {
        self.obs.get(self.step as i64).copy_(&obs);
        self.actions.get(self.step as i64).copy_(&action);
        self.logprobs.get(self.step as i64).copy_(&logprob);
        self.rewards.get(self.step as i64).copy_(&reward);
        self.dones.get(self.step as i64).copy_(&done);
        self.values.get(self.step as i64).copy_(&value);

        self.step += 1;
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.episode_returns.clear();
    }

    pub fn add_episode_return(&mut self, episode_return: f64) {
        self.episode_returns.push(episode_return);
    }
}
