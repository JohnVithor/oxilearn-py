use std::time::Instant;

use crate::{
    dqn::{DoubleDeepAgent, OptimizerEnum},
    env::{PyEnv, SpaceInfo},
    epsilon_greedy::{EpsilonGreedy, EpsilonUpdateStrategy},
    experience_buffer::RandomExperienceBuffer,
    generate_policy,
    trainer::{TrainResults, Trainer},
    ActivationFunction, OxiLearnErr,
};
use pyo3::{exceptions::PyTypeError, pyclass, pymethods, Py, PyAny, PyResult};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use tch::{nn, Device, Kind, Tensor};

#[pyclass]
pub struct DQNAgent {
    net_arch: Vec<(i64, String)>,
    last_activation: ActivationFunction,
    epsilon: f32,
    epsilon_decay: f32,
    memory_size: usize,
    min_memory_size: usize,
    lr: f64,
    discount_factor: f32,
    agent: Option<DoubleDeepAgent>,
}

impl DQNAgent {
    fn get_activation(id: &str) -> ActivationFunction {
        match id {
            "relu" => |xs: &Tensor| xs.relu(),
            "gelu" => |xs: &Tensor| xs.gelu("none"),
            "softmax" => |xs: &Tensor| xs.softmax(0, Kind::Float),
            _ => |xs: &Tensor| xs.shallow_clone(),
        }
    }
}

#[pymethods]
impl DQNAgent {
    #[new]
    #[pyo3(signature = (net_arch, last_activation="none", memory_size=5_000, min_memory_size=1_000, lr=0.0005, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.0005))]
    fn new(
        net_arch: Vec<(i64, String)>,
        last_activation: &str,
        memory_size: usize,
        min_memory_size: usize,
        lr: f64,
        discount_factor: f32,
        epsilon: f32,
        epsilon_decay: f32,
    ) -> Self {
        Self {
            net_arch,
            last_activation: DQNAgent::get_activation(last_activation),
            epsilon,
            epsilon_decay,
            memory_size,
            min_memory_size,
            lr,
            discount_factor,
            agent: None,
        }
    }

    #[pyo3(signature = (env, steps=200_000, update_freq=10, eval_at=50, eval_for=10))]
    pub fn train(
        &mut self,
        env: Py<PyAny>,
        steps: u128,
        update_freq: u128,
        eval_at: u128,
        eval_for: u128,
    ) -> PyResult<f32> {
        let mut rng: StdRng = StdRng::seed_from_u64(4);

        tch::manual_seed(rng.next_u64() as i64);
        tch::maybe_init_cuda();

        let env = PyEnv::new(env)?;

        let input = match env.observation_space().unwrap() {
            SpaceInfo::Discrete(_) => Err(PyTypeError::new_err("ambiente inválido")),
            SpaceInfo::Continuous(s) => Ok(s.len()),
        }? as i64;
        let output = match env.action_space().unwrap() {
            SpaceInfo::Discrete(n) => Ok(n),
            SpaceInfo::Continuous(_) => Err(PyTypeError::new_err("ambiente inválido")),
        }? as i64;

        let decay = self.epsilon_decay;

        let mem_replay = RandomExperienceBuffer::new(
            self.memory_size,
            self.min_memory_size,
            rng.next_u64(),
            Device::Cpu,
        );

        let action_selector = EpsilonGreedy::new(
            self.epsilon,
            rng.next_u64(),
            EpsilonUpdateStrategy::EpsilonDecreasing {
                final_epsilon: 0.0,
                epsilon_decay: Box::new(move |a| a - decay),
            },
        );

        let opt = OptimizerEnum::Adam(nn::Adam::default());

        let info = self
            .net_arch
            .iter()
            .map(|(n, func)| (*n, Self::get_activation(func)))
            .collect();

        let arch = generate_policy(info, self.last_activation, input, output).unwrap();

        let mut agent = DoubleDeepAgent::new(
            action_selector,
            mem_replay,
            arch,
            opt,
            self.lr,
            self.discount_factor,
            Device::Cpu,
        );

        let mut trainer = Trainer::new(env).unwrap();
        trainer.early_stop = Some(Box::new(|reward| reward >= 500.0));

        let start = Instant::now();

        let r: Result<TrainResults, OxiLearnErr> =
            trainer.train_by_steps(&mut agent, steps, update_freq, eval_at, eval_for);
        let elapsed = start.elapsed();
        println!("Elapsed time: {elapsed:?}");
        let rewards = r.unwrap().3;
        let reward_max = rewards
            .iter()
            .fold(rewards[0], |o, r| if *r > o { *r } else { o });
        self.agent = Some(agent);
        Ok(reward_max)
    }
}
