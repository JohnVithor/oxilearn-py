use rand::seq::SliceRandom;
use std::{collections::HashMap, time::Instant};
use tch::{
    nn::{self, Optimizer, OptimizerConfig},
    Device, Kind, Tensor,
};

use crate::{dqn::optimizer_enum::OptimizerEnum, env::PyEnv};

use super::{model::Policy, rollout::Rollout};

#[derive(Debug)]
pub struct OptimizationResults {
    pub learning_rate: f64,
    pub value_loss: f64,
    pub policy_loss: f64,
    pub entropy: f64,
    pub old_approx_kl: f64,
    pub approx_kl: f64,
    pub explained_variance: f64,
}

pub struct PPO {
    policy: Policy,
    optimizer: Optimizer,
    env: PyEnv,
    eval_env: PyEnv,
    num_steps: usize,
    rollout_buffer: Rollout,
    gamma: f64,
    learning_rate: f64,
    current_lr: f64,
    gae_lambda: f64,
    batch_size: usize,
    minibatch_size: usize,
    update_epochs: usize,
    clip_coef: f64,
    norm_adv: bool,
    clip_vloss: bool,
    ent_coef: f64,
    vf_coef: f64,
    max_grad_norm: f64,
    target_kl: Option<f64>,
    device: Device,
}

impl PPO {
    pub fn new(
        policy: Policy,
        optimizer: OptimizerEnum,
        env: PyEnv,
        eval_env: PyEnv,
        num_steps: usize,
        gamma: f64,
        learning_rate: f64,
        gae_lambda: f64,
        batch_size: usize,
        minibatch_size: usize,
        update_epochs: usize,
        clip_coef: f64,
        norm_adv: bool,
        clip_vloss: bool,
        ent_coef: f64,
        vf_coef: f64,
        max_grad_norm: f64,
        target_kl: Option<f64>,
        device: Device,
    ) -> Self {
        let rollout_buffer = Rollout::new(num_steps, &env, device);
        PPO {
            optimizer: optimizer.build(policy.varstore(), learning_rate).unwrap(),
            policy,
            env,
            eval_env,
            num_steps,
            rollout_buffer,
            gamma,
            learning_rate,
            current_lr: learning_rate,
            gae_lambda,
            batch_size,
            minibatch_size,
            update_epochs,
            clip_coef,
            norm_adv,
            clip_vloss,
            ent_coef,
            vf_coef,
            max_grad_norm,
            target_kl,
            device,
        }
    }

    pub fn collect_rollout(&mut self, mut next_obs: Tensor, mut next_done: bool) -> (Tensor, bool) {
        self.rollout_buffer.reset();

        for _ in 0..self.num_steps {
            let (new_next_obs, new_next_done) = self.collect_step(next_obs, next_done);
            next_obs = new_next_obs;
            next_done = new_next_done;
        }

        (next_obs, next_done)
    }

    fn collect_step(&mut self, next_obs: Tensor, next_done: bool) -> (Tensor, bool) {
        let obs = next_obs.shallow_clone();
        let done = next_done;

        let (action, logprob, _, value) = self.policy.get_action_and_value(&next_obs, None);
        let value = value.flatten(0, 1);

        let (new_obs, reward, terminations, truncations) =
            self.env.step(action.int64_value(&[0]) as usize).unwrap();
        let new_next_done = terminations | truncations;
        let new_next_obs = if new_next_done {
            self.env.reset(None).unwrap().to_device(self.device)
        } else {
            new_obs.to_device(self.device)
        };

        self.rollout_buffer
            .add(&obs, &action, &logprob, reward as f64, done, &value);

        // self.check_final_info(&infos);
        (new_next_obs, new_next_done)
    }

    fn compute_advantages_returns(
        &mut self,
        next_obs: &Tensor,
        next_done: bool,
    ) -> (Tensor, Tensor) {
        let advantages = self.compute_advantages(next_obs, next_done);
        let returns = &advantages + &self.rollout_buffer.values;
        (advantages, returns)
    }

    fn compute_advantages(&mut self, next_obs: &Tensor, next_done: bool) -> Tensor {
        let next_value = self.policy.get_value(next_obs).reshape(&[1, -1]);
        let advantages = Tensor::zeros_like(&self.rollout_buffer.rewards);
        let mut lastgaelam = 0.0;

        for t in (0..self.rollout_buffer.step).rev() {
            let id = t as i64;
            let nextnonterminal = if t == self.rollout_buffer.step - 1 {
                1.0 - next_done as i64 as f64
            } else {
                1.0 - self.rollout_buffer.dones.get(id + 1).double_value(&[0])
            };
            let mut nextvalues = &self.rollout_buffer.values.get(id + 1);
            if t == self.rollout_buffer.step - 1 {
                nextvalues = &next_value
            }
            let delta = &self.rollout_buffer.rewards.get(id)
                + self.gamma * nextvalues * nextnonterminal
                - &self.rollout_buffer.values.get(id);
            let advj = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam;
            advantages.get(id).copy_(&advj);
            lastgaelam = advj.double_value(&[0]);
        }

        advantages
    }

    fn anneal_learning_rate(&mut self, iteration: usize, num_iterations: usize) {
        let frac = 1.0 - ((iteration as f64 - 1.0) / num_iterations as f64);
        self.current_lr = frac * self.learning_rate;
        // self.optimizer.set_lr(self.current_lr);
    }

    fn flatten_data(
        &self,
        advantages: &Tensor,
        returns: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let b_obs = self
            .rollout_buffer
            .obs
            .view([-1, self.env.observation_space().unwrap().shape()]);
        let b_logprobs = self.rollout_buffer.logprobs.view([-1]);
        let b_actions = self
            .rollout_buffer
            .actions
            .view([-1, self.env.action_space().unwrap().shape()]);
        let b_advantages = advantages.view([-1]);
        let b_returns = returns.view([-1]);
        let b_values = self.rollout_buffer.values.view([-1]);
        (
            b_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
        )
    }

    fn optimize_epoch(
        &mut self,
        b_inds: &mut [i64],
        b_obs: &Tensor,
        b_actions: &Tensor,
        b_logprobs: &Tensor,
        b_returns: &Tensor,
        b_values: &Tensor,
        b_advantages: &Tensor,
    ) -> (f64, f64, f64, f64, f64) {
        b_inds.shuffle(&mut rand::thread_rng());

        let mut final_old_approx_kl = 0.0;
        let mut final_approx_kl = 0.0;
        let mut final_pg_loss = 0.0;
        let mut final_v_loss = 0.0;
        let mut final_entropy_loss = 0.0;

        for start in (0..self.batch_size).step_by(self.minibatch_size) {
            let end = start + self.minibatch_size;
            let mb_inds = &b_inds[start..end];

            let mb_obs = b_obs.index_select(0, &Tensor::from_slice(mb_inds));
            let mb_actions = b_actions.index_select(0, &Tensor::from_slice(mb_inds));
            let mb_logprobs = b_logprobs.index_select(0, &Tensor::from_slice(mb_inds));
            let mb_returns = b_returns.index_select(0, &Tensor::from_slice(mb_inds));
            let mb_values = b_values.index_select(0, &Tensor::from_slice(mb_inds));
            let mb_advantages = b_advantages.index_select(0, &Tensor::from_slice(mb_inds));

            let (old_approx_kl, approx_kl, pg_loss, v_loss, entropy_loss) = self
                .calculate_policy_loss(
                    &mb_obs,
                    &mb_actions,
                    &mb_logprobs,
                    &mb_returns,
                    &mb_values,
                    &mb_advantages,
                );
            final_old_approx_kl = old_approx_kl;
            final_approx_kl = approx_kl;
            final_pg_loss = pg_loss;
            final_v_loss = v_loss;
            final_entropy_loss = entropy_loss;
        }

        (
            final_old_approx_kl,
            final_approx_kl,
            final_pg_loss,
            final_v_loss,
            final_entropy_loss,
        )
    }

    pub fn optimize(&mut self, advantages: Tensor, returns: Tensor) -> OptimizationResults {
        let (b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values) =
            self.flatten_data(&advantages, &returns);
        let mut b_inds: Vec<i64> = (0..self.batch_size).map(|x| x as i64).collect();

        let mut old_approx_kl = 0.0;
        let mut approx_kl = 0.0;
        let mut pg_loss = 0.0;
        let mut v_loss = 0.0;
        let mut entropy_loss = 0.0;

        for _epoch in 0..self.update_epochs {
            let (
                epoch_old_approx_kl,
                epoch_approx_kl,
                epoch_pg_loss,
                epoch_v_loss,
                epoch_entropy_loss,
            ) = self.optimize_epoch(
                &mut b_inds,
                &b_obs,
                &b_actions,
                &b_logprobs,
                &b_returns,
                &b_values,
                &b_advantages,
            );

            old_approx_kl = epoch_old_approx_kl;
            approx_kl = epoch_approx_kl;
            pg_loss = epoch_pg_loss;
            v_loss = epoch_v_loss;
            entropy_loss = epoch_entropy_loss;

            if let Some(target_kl) = self.target_kl {
                if approx_kl > target_kl {
                    break;
                }
            }
        }

        let var_y = b_returns.var(true).double_value(&[0]);
        let explained_var = if var_y == 0.0 {
            std::f64::NAN
        } else {
            let diff = b_values - b_returns;
            1.0 - diff.var(true).double_value(&[0]) / var_y
        };

        OptimizationResults {
            learning_rate: self.current_lr,
            value_loss: v_loss,
            policy_loss: pg_loss,
            entropy: entropy_loss,
            old_approx_kl,
            approx_kl,
            explained_variance: explained_var,
        }
    }

    fn calculate_policy_loss(
        &mut self,
        mb_obs: &Tensor,
        mb_actions: &Tensor,
        mb_logprobs: &Tensor,
        mb_returns: &Tensor,
        mb_values: &Tensor,
        mb_advantages: &Tensor,
    ) -> (f64, f64, f64, f64, f64) {
        let (_new_action, newlogprob, entropy, newvalue) =
            self.policy.get_action_and_value(mb_obs, Some(mb_actions));
        let logratio = &newlogprob - mb_logprobs;
        let ratio = logratio.exp();

        let old_approx_kl = (-&logratio).mean(Kind::Float).double_value(&[0]);
        let approx_kl = ((&ratio - 1.0) - &logratio)
            .mean(Kind::Float)
            .double_value(&[0]);

        let mb_advantages = if self.norm_adv {
            (mb_advantages - mb_advantages.mean(Kind::Float)) / (mb_advantages.std(true) + 1e-8)
        } else {
            mb_advantages.shallow_clone()
        };

        let pg_loss1 = -&mb_advantages * &ratio;
        let pg_loss2 = -mb_advantages * ratio.clamp(1.0 - self.clip_coef, 1.0 + self.clip_coef);
        let pg_loss = pg_loss1.max_other(&pg_loss2).mean(Kind::Float);

        let newvalue = newvalue.view([-1]);
        let v_loss = if self.clip_vloss {
            let v_loss_unclipped = (&newvalue - mb_returns).pow(&Tensor::from(2));
            let v_clipped =
                mb_values + (&newvalue - mb_values).clamp(-self.clip_coef, self.clip_coef);
            let v_loss_clipped = (&v_clipped - mb_returns).pow(&Tensor::from(2));
            v_loss_unclipped
                .max_other(&v_loss_clipped)
                .mean(Kind::Float)
                .double_value(&[0])
        } else {
            (&newvalue - mb_returns)
                .pow(&Tensor::from(2))
                .mean(Kind::Float)
                .double_value(&[0])
        };

        let entropy_loss = entropy.mean(Kind::Float).double_value(&[0]);
        let loss = &pg_loss - self.ent_coef * entropy_loss + (v_loss / 2.0) * self.vf_coef;

        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.clip_grad_norm(self.max_grad_norm);
        self.optimizer.step();

        (
            old_approx_kl,
            approx_kl,
            pg_loss.double_value(&[0]),
            v_loss,
            entropy_loss,
        )
    }

    pub fn learn(
        &mut self,
        num_iterations: usize,
        seed: u64,
        anneal_lr: bool,
    ) -> Vec<OptimizationResults> {
        let start_time = Instant::now();
        let mut next_obs = self.env.reset(Some(seed)).unwrap().to_device(self.device);
        let mut next_done = false;

        let mut results = vec![];
        let mut checkpoint = num_iterations / 10;

        for global_step in (0..num_iterations).step_by(self.num_steps) {
            if anneal_lr {
                self.anneal_learning_rate(global_step, num_iterations);
            }
            let (new_next_obs, new_next_done) = self.collect_rollout(next_obs, next_done);
            next_obs = new_next_obs;
            next_done = new_next_done;

            let (advantages, returns) = self.compute_advantages_returns(&next_obs, next_done);
            let result = self.optimize(advantages, returns);
            results.push(result);

            if global_step >= checkpoint {
                checkpoint += num_iterations / 10;
                let evals = self.evaluate(10);
                let mean_epi_return: f32 = evals.iter().sum::<f32>() / evals.len() as f32;
                println!(
                    "step {}/{} mean: {}",
                    global_step, num_iterations, mean_epi_return
                );
                if self.eval_env.reward_threshold.is_some()
                    && mean_epi_return > self.eval_env.reward_threshold.unwrap()
                {
                    println!("Threshold reached!");
                    break;
                }
            }
        }

        let end_time = Instant::now();
        println!(
            "Learning took {} seconds",
            (end_time - start_time).as_secs_f64()
        );
        results
    }

    pub fn evaluate(&mut self, num_episodes: usize) -> Vec<f32> {
        let mut rewards = vec![];

        for _ in 0..num_episodes {
            let mut obs = self.eval_env.reset(None).unwrap().to_device(self.device);
            let mut done = false;
            let mut episode_reward = 0.0;

            while !done {
                let action = self.policy.get_best_action(&obs);
                let (new_obs, reward, terminations, truncations) = self
                    .eval_env
                    .step(action.int64_value(&[0]) as usize)
                    .unwrap();
                done = terminations || truncations;
                obs = new_obs.to_device(self.device);
                episode_reward += reward;
            }

            rewards.push(episode_reward);
        }

        rewards
    }
}
