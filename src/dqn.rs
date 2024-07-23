
use oxilearn::{dqn::{agent::{DQNAgent, ParametersDQN, TrainResults}, epsilon_greedy::{EpsilonGreedy, EpsilonUpdateStrategy}, experience_buffer::RandomExperienceBuffer, losses::{huber, mae, mse, rmse, smooth_l1}, policy::{generate_policy, ActivationFunction}}, env::{Env, SpaceInfo}, optimizer_enum::OptimizerEnum, OxiLearnErr};
use pyo3::{
  exceptions::{PyFileNotFoundError, PyTypeError, PyValueError},
  pyclass, pymethods, Bound, PyAny, PyResult, Python,
};
use rand::{rngs::SmallRng, RngCore, SeedableRng};
use tch::{
  nn::{Adam, AdamW, RmsProp, Sgd},
  Device, Kind, Tensor,
};

use crate::env::PyEnv;

#[pyclass]
pub struct DQN {
  net_arch: Vec<(i64, String)>,
  last_activation: ActivationFunction,
  initial_epsilon: f32,
  final_epsilon: f32,
  exploration_fraction: f32,
  memory_size: i64,
  min_memory_size: i64,
  learning_rate: f64,
  discount_factor: f32,
  max_grad_norm: f64,
  gradient_steps: u32,
  train_freq: u32,
  batch_size: usize,
  update_freq: u32,
  eval_freq: u32,
  eval_for: u32,
  agent: Option<DQNAgent>,
  rng: SmallRng,
  normalize_obs: bool,
  optimizer: OptimizerEnum,
  loss_fn: fn(&Tensor, &Tensor) -> Tensor,
}

impl DQN {
  fn get_activation(id: &str) -> ActivationFunction {
      match id {
          "relu" => |xs: &Tensor| xs.relu(),
          "gelu" => |xs: &Tensor| xs.gelu("none"),
          "softmax" => |xs: &Tensor| xs.softmax(0, Kind::Float),
          "tanh" => |xs: &Tensor| xs.tanh(),
          _ => |xs: &Tensor| xs.shallow_clone(),
      }
  }
}

#[pymethods]
impl DQN {
  /// Create a new DQNAgent
  #[new]
  #[pyo3(signature = (
      net_arch,
      learning_rate,
      last_activation="none",
      memory_size=5_000,
      min_memory_size=1_000,
      discount_factor=0.99,
      initial_epsilon=1.0,
      final_epsilon=0.05,
      exploration_fraction=0.05,
      max_grad_norm=10.0,
      gradient_steps=1,
      train_freq=1,
      batch_size=32,
      update_freq=10,
      eval_freq=1_000,
      eval_for=10,
      seed=0,
      normalize_obs=false,
      optimizer="Adam",
      // optimizer_info=HashMap::default(),
      loss_fn="MSE"
  ))]
  #[allow(clippy::too_many_arguments)]
  fn new(
      net_arch: Vec<(i64, String)>,
      learning_rate: f64,
      last_activation: &str,
      memory_size: i64,
      min_memory_size: i64,
      discount_factor: f32,
      initial_epsilon: f32,
      final_epsilon: f32,
      exploration_fraction: f32,
      max_grad_norm: f64,
      gradient_steps: u32,
      train_freq: u32,
      batch_size: usize,
      update_freq: u32,
      eval_freq: u32,
      eval_for: u32,
      seed: u64,
      normalize_obs:bool,
      optimizer: &str,
      // optimizer_info: HashMap<String, String>,
      loss_fn: &str,
      py: Python,
  ) -> PyResult<Self> {
      py.allow_threads(|| -> PyResult<Self> {
          let mut rng: SmallRng = SmallRng::seed_from_u64(seed);
          tch::manual_seed(rng.next_u64() as i64);
        //   tch::maybe_init_cuda();
  
          let optimizer_info = match optimizer {
              "Adam" => Some(OptimizerEnum::Adam(Adam::default())),
              "Sgd" => Some(OptimizerEnum::Sgd(Sgd::default())),
              "RmsProp" => Some(OptimizerEnum::RmsProp(RmsProp::default())),
              "AdamW" => Some(OptimizerEnum::AdamW(AdamW::default())),
              _ => None,
          };
  
          let Some(optimizer) = optimizer_info else {
              return Err(PyValueError::new_err(format!(
                  "Invalid optimizer option '{optimizer}' valid options are: 'Adam', 'Sgd', 'RmsProp' and 'AdamW'"
              )));
          };
  
          let loss_fn_info: Option<fn(&Tensor, &Tensor) -> Tensor> = match loss_fn {
              "MAE" => Some(mae),
              "MSE" => Some(mse),
              "RMSE" => Some(rmse),
              "Huber" => Some(huber),
              "smooth_l1" => Some(smooth_l1),
              _ => None,
          };
  
          let Some(loss_fn) = loss_fn_info else {
              return Err(PyValueError::new_err(format!(
                  "Invalid loss_fn option '{loss_fn}' valid options are: 'MAE','MSE', 'RMSE', 'Huber' and smooth_l1"
              )));
          };
  
          Ok(Self {
              net_arch,
              last_activation: DQN::get_activation(last_activation),
              initial_epsilon,
              final_epsilon,
              exploration_fraction,
              memory_size,
              min_memory_size,
              learning_rate,
              discount_factor,
              max_grad_norm,
              gradient_steps,
              train_freq,
              batch_size,
              update_freq,
              eval_freq,
              eval_for,
              agent: None,
              rng,
              optimizer,
              normalize_obs,
              loss_fn,
          })})
  }

  #[pyo3(signature = ())]
  pub fn reset(&mut self) {
      self.agent = None;
  }

  #[pyo3(signature = (path))]
  pub fn save(&self, path: &str) -> PyResult<()> {
      if let Some(agent) = &self.agent {
          match agent.save_net(path) {
              Ok(_) => Ok(()),
              Err(e) => Err(PyFileNotFoundError::new_err(e.to_string())),
          }
      } else {
          Err(PyValueError::new_err("Agent not initialized!"))
      }
  }

  #[pyo3(signature = (path))]
  pub fn load(&mut self, path: &str) -> PyResult<()> {
      if let Some(agent) = &mut self.agent {
          match agent.load_net(path) {
              Ok(_) => Ok(()),
              Err(e) => Err(PyFileNotFoundError::new_err(e.to_string())),
          }
      } else {
          Err(PyValueError::new_err("Agent not initialized!"))
      }
  }

  #[pyo3(signature = (environment))]
  pub fn prepare(&mut self, environment: Bound<PyAny>) -> PyResult<()> {
      let environment = PyEnv::new(environment)?;
      self.create_agent(&environment)?;
      Ok(())
  }

  fn create_agent(&mut self, environment: &PyEnv) -> PyResult<()> {
      let input = match environment.observation_space().unwrap() {
          SpaceInfo::Discrete(_) => Err(PyTypeError::new_err("ambiente inválido")),
          SpaceInfo::Continuous(s) => Ok(s.len()),
      }? as i64;
      let output = match environment.action_space().unwrap() {
          SpaceInfo::Discrete(n) => Ok(n),
          SpaceInfo::Continuous(_) => Err(PyTypeError::new_err("ambiente inválido")),
      }? as i64;
      let mem_replay = RandomExperienceBuffer::new(
          self.memory_size,
          input,
          self.min_memory_size,
          self.rng.next_u64(),
          self.normalize_obs,
          Device::cuda_if_available(),
      );

      let action_selector = EpsilonGreedy::new(
          self.initial_epsilon,
          self.rng.next_u64(),
          EpsilonUpdateStrategy::EpsilonLinearTrainingDecreasing {
              start: self.initial_epsilon,
              end: self.final_epsilon,
              end_fraction: self.exploration_fraction,
          },
      );

      let info = self
          .net_arch
          .iter()
          .map(|(n, func)| (*n, Self::get_activation(func)))
          .collect();

      let arch = generate_policy(info, self.last_activation, input, output).unwrap();
      let parameters = ParametersDQN {
          learning_rate: self.learning_rate,
          discount_factor: self.discount_factor,
          max_grad_norm: self.max_grad_norm,
          gradient_steps: self.gradient_steps,
          train_freq: self.train_freq,
          batch_size: self.batch_size,
          update_freq: self.update_freq,
          eval_freq: self.eval_freq,
          eval_for: self.eval_for,
      };
      self.agent = Some(DQNAgent::new(
          action_selector,
          mem_replay,
          arch,
          self.optimizer,
          self.loss_fn,
          parameters,
          Device::cuda_if_available(),
      ));
      Ok(())
  }

  #[pyo3(signature = (
      env,
      eval_env,
      steps=200_000,
      verbose=0
  ))]
  
  #[allow(clippy::too_many_arguments)]
  pub fn train(
      &mut self,
      env: Bound<PyAny>,
      eval_env: Bound<PyAny>,
      steps: u32,
      verbose: usize,
      py: Python,
  ) -> PyResult<TrainResults> {
      let mut env = PyEnv::new(env)?;
      let mut eval_env = PyEnv::new(eval_env)?;
      if self.agent.is_none() {
          self.create_agent(&env)?;
      }
      ;
      py.allow_threads(|| ->PyResult<TrainResults> {
          let r: Result<TrainResults, OxiLearnErr> = self.agent.as_mut().unwrap().train_by_steps(
            &mut env, &mut eval_env,
              steps,
              verbose,
          );
          Ok(r.unwrap())
      })
  }

  #[pyo3(signature = (env, n_eval_episodes))]
  #[allow(clippy::too_many_arguments)]
  pub fn evaluate(&mut self, env: Bound<PyAny>, n_eval_episodes: u32, py: Python) -> PyResult<(f32, f32)> {
      let mut eval_env = PyEnv::new(env)?;
      if self.agent.is_none() {
          self.create_agent(&eval_env)?;
      }
      py.allow_threads(|| -> PyResult<(f32, f32)> {
          let r = self.agent.as_mut().unwrap().evaluate(&mut eval_env, n_eval_episodes);
          let rewards = r.unwrap().0;
          let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
          let variance = rewards
              .iter()
              .map(|value| {
                  let diff = reward_avg - *value;
                  diff * diff
              })
              .sum::<f32>()
              / rewards.len() as f32;
          Ok((reward_avg, variance.sqrt()))
      })
  }
}