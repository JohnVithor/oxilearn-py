use tch::{Device, Tensor};

use crate::env::PyEnv;

pub struct Rollout {
    pub obs: Tensor,
    pub actions: Tensor,
    pub logprobs: Tensor,
    pub rewards: Tensor,
    pub dones: Tensor,
    pub values: Tensor,
    pub episode_returns: Vec<f64>,
    pub step: usize,
}

impl Rollout {
    pub fn new(num_steps: usize, env: &PyEnv, device: Device) -> Self {
        let obs_size = env.observation_space().unwrap().shape();
        let action_size = env.action_space().unwrap().shape();

        let capacity = num_steps as i64;
        Self {
            obs: Tensor::empty([capacity, obs_size], (tch::Kind::Double, device)),
            actions: Tensor::empty([capacity, action_size], (tch::Kind::Double, device)),
            logprobs: Tensor::empty(capacity, (tch::Kind::Double, device)),
            rewards: Tensor::empty(capacity, (tch::Kind::Double, device)),
            dones: Tensor::empty(capacity, (tch::Kind::Int8, device)),
            values: Tensor::empty(capacity, (tch::Kind::Double, device)),
            episode_returns: Vec::new(),
            step: 0,
        }
    }

    pub fn add(
        &mut self,
        obs: &Tensor,
        action: &Tensor,
        logprob: &Tensor,
        reward: f64,
        done: bool,
        value: &Tensor,
    ) {
        self.obs.get(self.step as i64).copy_(obs);
        self.actions.get(self.step as i64).copy_(action);
        self.logprobs.get(self.step as i64).copy_(logprob);
        self.rewards.get(self.step as i64).fill_(reward);
        self.dones.get(self.step as i64).fill_(done as i64);
        self.values.get(self.step as i64).copy_(value);

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
