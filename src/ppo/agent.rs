use ndarray::{array, Array, Axis};
use std::collections::HashMap;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device, Tensor};

use crate::env::Env;

use super::{model::Policy, rollout::Rollout};

#[derive(Debug)]
pub struct OptimizationResults {
    pub learning_rate: f64,
    pub value_loss: f64,
    pub policy_loss: f64,
    pub entropy: f64,
    pub old_approx_kl: f64,
    pub approx_kl: f64,
    pub clipfrac: f64,
    pub explained_variance: f64,
}

pub struct PPO {
    policy: Policy,
    optimizer: nn::Adam,
    envs: Env,
    eval_env: Env,
    num_steps: usize,
    num_envs: usize,
    gamma: f64,
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
    rollout_buffer: Rollout,
    learning_rate: f64,
}

impl PPO {
    pub fn new(
        policy: Policy,
        optimizer: nn::Adam,
        envs: Env,
        eval_env: Env,
        num_steps: usize,
        num_envs: usize,
        gamma: f64,
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
        let rollout_buffer = Rollout::new(num_steps, num_envs, &envs, device);
        let learning_rate = optimizer.lr();

        PPO {
            policy,
            optimizer,
            envs,
            eval_env,
            num_steps,
            num_envs,
            gamma,
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
            rollout_buffer,
            learning_rate,
        }
    }

    pub fn collect_rollout(
        &mut self,
        global_step: &mut usize,
        mut next_obs: Tensor,
        mut next_done: Tensor,
    ) -> (Tensor, Tensor, usize) {
        self.rollout_buffer.reset();

        for _ in 0..self.num_steps {
            *global_step += self.num_envs;
            let obs_step = next_obs.copy();
            let dones_step = next_done.copy();

            let (action, logprob, _, value) = self.policy.get_action_and_value(&next_obs, None);
            let values_step = value.flatten(0, 1);
            let actions_step = action.copy();
            let logprobs_step = logprob.copy();

            let (obs, reward, terminations, truncations, infos) =
                self.envs.step(&action.cpu().to_ndarray()).unwrap();
            next_done = terminations.bitor(&truncations).to_tch(&self.device);
            next_obs = obs.to_tch(&self.device);
            let rewards_step = reward.to_tch(&self.device).view(-1);

            self.rollout_buffer.add(
                &obs_step,
                &actions_step,
                &logprobs_step,
                &rewards_step,
                &dones_step,
                &values_step,
            );

            if let Some(final_info) = infos.get("final_info") {
                for info in final_info.iter() {
                    if let Some(episode) = info.get("episode") {
                        if let Some(r) = episode.get("r") {
                            self.rollout_buffer.add_episode_return(*r);
                        }
                    }
                }
            }
        }

        (next_obs, next_done, *global_step)
    }

    pub fn compute_advantages_returns(
        &self,
        next_obs: &Tensor,
        next_done: &Tensor,
    ) -> (Tensor, Tensor) {
        let mut advantages = Tensor::zeros_like(&self.rollout_buffer.rewards);
        let mut lastgaelam = 0.0;

        let next_value = self.policy.get_value(next_obs).view_1d();
        for t in (0..self.rollout_buffer.step).rev() {
            let nextnonterminal = if t == self.rollout_buffer.step - 1 {
                1.0 - next_done
            } else {
                1.0 - self.rollout_buffer.dones.get(t + 1).view_1d()
            };
            let nextvalues = if t == self.rollout_buffer.step - 1 {
                next_value
            } else {
                self.rollout_buffer.values.get(t + 1).view_1d()
            };
            let delta = self.rollout_buffer.rewards.get(t)
                + self.gamma * nextvalues * nextnonterminal
                - self.rollout_buffer.values.get(t);
            advantages.set(
                t,
                &((delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam).view_1d()),
            );
            lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam;
        }
        let returns = &advantages + &self.rollout_buffer.values;
        (advantages, returns)
    }

    pub fn anneal_learning_rate(&mut self, iteration: usize, num_iterations: usize) {
        let frac = 1.0 - (iteration as f64 - 1.0) / num_iterations as f64;
        self.optimizer.set_lr(frac * self.learning_rate);
    }

    pub fn _flatten_data(
        &self,
        advantages: &Tensor,
        returns: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let b_obs = self.rollout_buffer.obs.view(&[
            -1,
            self.envs
                .observation_space()
                .unwrap()
                .dim()
                .iter()
                .product::<i64>(),
        ]);
        let b_logprobs = self.rollout_buffer.logprobs.view(&[-1]);
        let b_actions = self.rollout_buffer.actions.view(&[
            -1,
            self.envs
                .action_space()
                .unwrap()
                .dim()
                .iter()
                .product::<i64>(),
        ]);
        let b_advantages = advantages.view(&[-1]);
        let b_returns = returns.view(&[-1]);
        let b_values = self.rollout_buffer.values.view(&[-1]);
        (
            b_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
        )
    }

    pub fn optimize(&mut self, advantages: Tensor, returns: Tensor) -> OptimizationResults {
        let (b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values) =
            self._flatten_data(&advantages, &returns);
        let b_inds: Vec<usize> = (0..self.batch_size).collect();
        let mut clipfracs = vec![];

        for _ in 0..self.update_epochs {
            let mut indices = b_inds.clone();
            indices.shuffle(&mut rand::thread_rng());
            for start in (0..self.batch_size).step_by(self.minibatch_size) {
                let end = start + self.minibatch_size;
                let mb_inds = &indices[start..end];

                let (newlogprob, entropy, newvalue) = self.policy.get_action_and_value(
                    &b_obs.index_select(0, &mb_inds.into()),
                    Some(b_actions.index_select(0, &mb_inds.into())),
                );

                let logratio = &newlogprob - &b_logprobs.index_select(0, &mb_inds.into());
                let ratio = logratio.exp();

                let old_approx_kl = -logratio.mean(tch::Kind::Float).unwrap();
                let approx_kl = (&ratio - 1.0).powi(2).mean(tch::Kind::Float).unwrap();
                clipfracs.push(
                    (ratio.abs() > self.clip_coef)
                        .float()
                        .mean(tch::Kind::Float)
                        .unwrap(),
                );

                let mut mb_advantages = b_advantages.index_select(0, &mb_inds.into());
                if self.norm_adv {
                    mb_advantages = (&mb_advantages
                        - mb_advantages.mean(tch::Kind::Float).unwrap())
                        / (mb_advantages.std(tch::Kind::Float).unwrap() + 1e-8);
                }

                let pg_loss1 = -&mb_advantages * &ratio;
                let pg_loss2 =
                    -&mb_advantages * ratio.clamp(1.0 - self.clip_coef, 1.0 + self.clip_coef);
                let pg_loss = pg_loss1.max(pg_loss2).mean(tch::Kind::Float).unwrap();

                let newvalue = newvalue.view_1d();
                let v_loss = if self.clip_vloss {
                    let v_loss_unclipped =
                        (newvalue - b_returns.index_select(0, &mb_inds.into())).powi(2);
                    let v_clipped = &b_values.index_select(0, &mb_inds.into())
                        + (newvalue - &b_values.index_select(0, &mb_inds.into()))
                            .clamp(-self.clip_coef, self.clip_coef);
                    let v_loss_clipped =
                        (v_clipped - b_returns.index_select(0, &mb_inds.into())).powi(2);
                    v_loss_unclipped
                        .max(v_loss_clipped)
                        .mean(tch::Kind::Float)
                        .unwrap()
                } else {
                    (newvalue - b_returns.index_select(0, &mb_inds.into()))
                        .powi(2)
                        .mean(tch::Kind::Float)
                        .unwrap()
                };

                let entropy_loss = entropy.mean(tch::Kind::Float).unwrap();
                let loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef;

                self.optimizer.zero_grad();
                loss.backward();
                self.policy.clip_grad_norm(self.max_grad_norm).unwrap();
                self.optimizer.step();
            }

            if let Some(target_kl) = self.target_kl {
                if approx_kl > target_kl {
                    break;
                }
            }
        }

        let y_pred = b_values.to_ndarray();
        let y_true = b_returns.to_ndarray();
        let var_y = y_true.var_axis(Axis(0), false);
        let explained_var = if var_y == 0.0 {
            f64::NAN
        } else {
            1.0 - (y_true - y_pred).var_axis(Axis(0), false) / var_y
        };

        OptimizationResults {
            learning_rate: self.optimizer.lr(),
            value_loss: v_loss,
            policy_loss: pg_loss,
            entropy: entropy_loss,
            old_approx_kl,
            approx_kl,
            clipfrac: clipfracs.mean_axis(Axis(0)).unwrap(),
            explained_variance: explained_var,
        }
    }

    pub fn learn(
        &mut self,
        num_iterations: usize,
        seed: u64,
        anneal_lr: bool,
    ) -> Vec<OptimizationResults> {
        let mut global_step = 0;
        let start_time = Instant::now();
        let (mut next_obs, _) = self.envs.reset(Some(seed)).unwrap();
        let next_done = Tensor::zeros(&[self.num_envs as i64], (tch::Kind::Float, self.device));

        let mut results = vec![];
        let mut checkpoint = 10_000;

        while global_step < num_iterations {
            if anneal_lr {
                self.anneal_learning_rate(global_step, num_iterations);
            }

            let (next_obs, next_done, global_step) =
                self.collect_rollout(&mut global_step, next_obs, next_done);
            let (advantages, returns) = self.compute_advantages_returns(&next_obs, &next_done);
            let result = self.optimize(advantages, returns);
            results.push(result);

            if global_step >= checkpoint {
                checkpoint += 10_000;
                let mean_epi_return = self.evaluate().mean().unwrap();
                println!(
                    "step {}/{}, mean: {}",
                    global_step, num_iterations, mean_epi_return
                );
                if mean_epi_return > self.eval_env.spec().unwrap().reward_threshold().unwrap() {
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

    pub fn evaluate(&mut self, num_episodes: usize) -> Vec<f64> {
        let mut rewards = vec![];

        for _ in 0..num_episodes {
            let (mut obs, _) = self.eval_env.reset(None).unwrap();
            let mut done = false;
            let mut episode_reward = 0.0;

            while !done {
                let action = self
                    .policy
                    .get_best_action(&obs.to_tch(&self.device))
                    .unwrap();
                let (next_obs, reward, terminations, truncations, _) =
                    self.eval_env.step(&action.cpu().to_ndarray()).unwrap();
                done = terminations || truncations;
                obs = next_obs;
                episode_reward += reward;
            }
            rewards.push(episode_reward);
        }

        rewards
    }
}
