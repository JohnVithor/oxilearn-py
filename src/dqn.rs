use crate::{
    epsilon_greedy::EpsilonGreedy, experience_buffer::RandomExperienceBuffer, PolicyGenerator,
};
use std::fs;
use tch::{
    nn::{Adam, AdamW, Module, Optimizer, OptimizerConfig, RmsProp, Sgd, VarStore},
    COptimizer, Device, Kind, TchError, Tensor,
};

pub enum OptimizerEnum {
    Adam(Adam),
    _Sgd(Sgd),
    _RmsProp(RmsProp),
    _AdamW(AdamW),
}

impl OptimizerConfig for OptimizerEnum {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        match self {
            OptimizerEnum::Adam(opt) => opt.build_copt(lr),
            OptimizerEnum::_Sgd(opt) => opt.build_copt(lr),
            OptimizerEnum::_RmsProp(opt) => opt.build_copt(lr),
            OptimizerEnum::_AdamW(opt) => opt.build_copt(lr),
        }
    }
}

pub struct DoubleDeepAgent {
    pub action_selection: EpsilonGreedy,
    pub policy: Box<dyn Module>,
    pub target_policy: Box<dyn Module>,
    pub policy_vs: VarStore,
    pub target_policy_vs: VarStore,
    pub optimizer: Optimizer,
    pub memory: RandomExperienceBuffer,
    pub discount_factor: f32,
}

impl DoubleDeepAgent {
    pub fn new(
        action_selector: EpsilonGreedy,
        mem_replay: RandomExperienceBuffer,
        generate_policy: Box<PolicyGenerator>,
        opt: OptimizerEnum,
        lr: f64,
        discount_factor: f32,
        device: Device,
    ) -> Self {
        let (policy_net, mem_policy) = generate_policy("q_net", device);
        let (target_net, mut mem_target) = generate_policy("q_net", device);
        mem_target.copy(&mem_policy).unwrap();
        Self {
            optimizer: opt.build(&mem_policy, lr).unwrap(),
            action_selection: action_selector,
            memory: mem_replay,
            policy: policy_net,
            policy_vs: mem_policy,
            target_policy: target_net,
            target_policy_vs: mem_target,
            discount_factor,
        }
    }

    pub fn get_action(&mut self, state: &Tensor) -> usize {
        let values = tch::no_grad(|| self.policy.forward(state));
        self.action_selection.get_action(&values) as usize
    }

    pub fn get_best_action(&self, state: &Tensor) -> usize {
        let values = tch::no_grad(|| self.policy.forward(state));
        let a: i32 = values.argmax(0, true).try_into().unwrap();
        a as usize
    }

    pub fn add_transition(
        &mut self,
        curr_state: &Tensor,
        curr_action: usize,
        reward: f32,
        done: bool,
        next_state: &Tensor,
    ) {
        self.memory
            .add(curr_state, curr_action, reward, done, next_state);
    }

    pub fn update_networks(&mut self) -> Result<(), TchError> {
        self.target_policy_vs.copy(&self.policy_vs)
    }

    pub fn get_batch(&mut self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        self.memory.sample_batch(size)
    }

    pub fn batch_qvalues(&self, b_states: &Tensor, b_actions: &Tensor) -> Tensor {
        self.policy.forward(b_states).gather(1, b_actions, false)
    }

    pub fn batch_expected_values(
        &self,
        b_state_: &Tensor,
        b_reward: &Tensor,
        b_done: &Tensor,
    ) -> Tensor {
        let best_target_qvalues =
            tch::no_grad(|| self.target_policy.forward(b_state_).max_dim(1, true).0);
        b_reward + self.discount_factor * (&Tensor::from(1.0) - b_done) * (&best_target_qvalues)
    }

    pub fn optimize(&mut self, loss: Tensor) {
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
    }

    pub fn update(&mut self) -> Option<f32> {
        if self.memory.ready() {
            let (b_state, b_action, b_reward, b_done, b_state_) = self.get_batch(32);
            let policy_qvalues = self.batch_qvalues(&b_state, &b_action);
            let expected_values = self.batch_expected_values(&b_state_, &b_reward, &b_done);
            let loss = policy_qvalues.mse_loss(&expected_values, tch::Reduction::Mean);
            self.optimize(loss);
            Some(expected_values.mean(Kind::Float).try_into().unwrap())
        } else {
            None
        }
    }

    pub fn action_selection_update(&mut self, epi_reward: f32) {
        self.action_selection.update(epi_reward);
    }

    pub fn _get_epsilon(&self) -> f32 {
        self.action_selection._get_epsilon()
    }

    pub fn reset(&mut self) {
        self.action_selection.reset();
        // TODO: reset policies
    }

    pub fn save_net(&self, path: &str) -> Result<(), TchError> {
        fs::create_dir_all(path)?;
        self.policy_vs
            .save(format!("{path}/policy_weights.safetensors"))?;
        self.target_policy_vs
            .save(format!("{path}/target_policy_weights.safetensors"))?;
        Ok(())
    }

    pub fn load_net(&mut self, path: &str) -> Result<(), TchError> {
        self.policy_vs
            .load(format!("{path}/policy_weights.safetensors"))?;
        self.target_policy_vs
            .load(format!("{path}/target_policy_weights.safetensors"))?;
        Ok(())
    }
}
