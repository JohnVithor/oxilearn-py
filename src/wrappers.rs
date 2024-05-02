use crate::{
    dqn::{huber, mae, mse, rmse, smooth_l1, DoubleDeepAgent, OptimizerEnum},
    env::{PyEnv, SpaceInfo},
    epsilon_greedy::{EpsilonGreedy, EpsilonUpdateStrategy},
    experience_buffer::RandomExperienceBuffer,
    generate_policy,
    trainer::{TrainResults, Trainer},
    ActivationFunction, OxiLearnErr,
};
use pyo3::{
    exceptions::{PyFileNotFoundError, PyTypeError, PyValueError},
    pyclass, pymethods, Bound, PyAny, PyResult, Python,
};
use rand::{rngs::SmallRng, RngCore, SeedableRng};
use tch::{
    nn::{Adam, AdamW, RmsProp, Sgd},
    Device, Kind, Tensor,
};

#[pyclass]
pub struct DQNAgent {
    net_arch: Vec<(i64, String)>,
    last_activation: ActivationFunction,
    initial_epsilon: f32,
    final_epsilon: f32,
    exploration_fraction: f32,
    memory_size: usize,
    min_memory_size: usize,
    lr: f64,
    discount_factor: f32,
    max_grad_norm: f64,
    agent: Option<DoubleDeepAgent>,
    rng: SmallRng,
    optimizer: OptimizerEnum,
    loss_fn: fn(&Tensor, &Tensor) -> Tensor,
}

impl DQNAgent {
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
impl DQNAgent {
    /// Create a new DQNAgent
    #[new]
    #[pyo3(signature = (
        net_arch,
        last_activation="none",
        memory_size=5_000,
        min_memory_size=1_000,
        lr=0.0005,
        discount_factor=0.99,
        initial_epsilon=1.0,
        final_epsilon=0.05,
        exploration_fraction=0.05,
        max_grad_norm=10.0,
        seed=0,
        optimizer="Adam",
        // optimizer_info=HashMap::default(),
        loss_fn="Huber"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        net_arch: Vec<(i64, String)>,
        last_activation: &str,
        memory_size: usize,
        min_memory_size: usize,
        lr: f64,
        discount_factor: f32,
        initial_epsilon: f32,
        final_epsilon: f32,
        exploration_fraction: f32,
        max_grad_norm: f64,
        seed: u64,
        optimizer: &str,
        // optimizer_info: HashMap<String, String>,
        loss_fn: &str,
    ) -> PyResult<Self> {
        let mut rng: SmallRng = SmallRng::seed_from_u64(seed);
        tch::manual_seed(rng.next_u64() as i64);
        tch::maybe_init_cuda();

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
            last_activation: DQNAgent::get_activation(last_activation),
            initial_epsilon,
            final_epsilon,
            exploration_fraction,
            memory_size,
            min_memory_size,
            lr,
            discount_factor,
            max_grad_norm,
            agent: None,
            rng,
            optimizer,
            loss_fn,
        })
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

    #[pyo3(signature = (env))]
    pub fn prepare(&mut self, env: Bound<PyAny>, py: Python<'_>) -> PyResult<()> {
        let env = PyEnv::new(env)?;
        self.create_agent(&env, py)?;
        Ok(())
    }

    fn create_agent(&mut self, env: &PyEnv, py: Python<'_>) -> PyResult<()> {
        let input = match env.observation_space(py).unwrap() {
            SpaceInfo::Discrete(_) => Err(PyTypeError::new_err("ambiente inválido")),
            SpaceInfo::Continuous(s) => Ok(s.len()),
        }? as i64;
        let output = match env.action_space(py).unwrap() {
            SpaceInfo::Discrete(n) => Ok(n),
            SpaceInfo::Continuous(_) => Err(PyTypeError::new_err("ambiente inválido")),
        }? as i64;
        self.agent = py.allow_threads(|| {
            let mem_replay = RandomExperienceBuffer::new(
                self.memory_size,
                self.min_memory_size,
                self.rng.next_u64(),
                Device::Cpu,
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
            Some(DoubleDeepAgent::new(
                action_selector,
                mem_replay,
                arch,
                self.optimizer,
                self.loss_fn,
                self.lr,
                self.discount_factor,
                self.max_grad_norm,
                Device::Cpu,
            ))
        });
        Ok(())
    }

    #[pyo3(signature = (env, solve_with, steps=200_000, update_freq=10, eval_at=50, eval_for=10, verbose=0))]
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        env: Bound<PyAny>,
        solve_with: f32,
        steps: u128,
        update_freq: u128,
        eval_at: u128,
        eval_for: u128,
        verbose: usize,
        py: Python<'_>,
    ) -> PyResult<TrainResults> {
        let env = PyEnv::new(env)?;
        if self.agent.is_none() {
            self.create_agent(&env, py)?;
        }

        py.allow_threads(|| {
            let mut trainer = Trainer::new(env).unwrap();
            trainer.early_stop = Some(Box::new(move |reward| reward >= solve_with));
            let r: Result<TrainResults, OxiLearnErr> = trainer.train_by_steps(
                self.agent.as_mut().unwrap(),
                steps,
                update_freq,
                eval_at,
                eval_for,
                verbose,
            );
            Ok(r.unwrap())
        })
    }

    #[pyo3(signature = (env, n_eval_episodes))]
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &mut self,
        env: Bound<PyAny>,
        n_eval_episodes: u128,
        py: Python<'_>,
    ) -> PyResult<(f32, f32)> {
        let env = PyEnv::new(env)?;
        if self.agent.is_none() {
            self.create_agent(&env, py)?;
        }

        py.allow_threads(|| {
            let mut trainer = Trainer::new(env).unwrap();
            let r = trainer.evaluate(self.agent.as_mut().unwrap(), n_eval_episodes);
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
