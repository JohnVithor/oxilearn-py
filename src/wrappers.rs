use crate::{
    dqn::{DoubleDeepAgent, OptimizerEnum},
    env::{PyEnv, SpaceInfo},
    epsilon_greedy::{EpsilonGreedy, EpsilonUpdateStrategy},
    experience_buffer::RandomExperienceBuffer,
    generate_policy,
    trainer::{TrainResults, Trainer},
    ActivationFunction, OxiLearnErr,
};
use pyo3::{exceptions::PyTypeError, pyclass, pymethods, Py, PyAny, PyResult, Python};
use rand::{rngs::SmallRng, RngCore, SeedableRng};
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
    rng: SmallRng,
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
    #[allow(clippy::too_many_arguments)]
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
        let mut rng: SmallRng = SmallRng::seed_from_u64(4);
        tch::manual_seed(rng.next_u64() as i64);
        tch::maybe_init_cuda();
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
            rng,
        }
    }

    #[pyo3(signature = ())]
    pub fn reset(&mut self) {
        self.agent = None;
    }

    #[pyo3(signature = (env, solve_with, steps=200_000, update_freq=10, eval_at=50, eval_for=10))]
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        env: Py<PyAny>,
        solve_with: f32,
        steps: u128,
        update_freq: u128,
        eval_at: u128,
        eval_for: u128,
        py: Python<'_>,
    ) -> PyResult<f32> {
        let env = PyEnv::new(env, py)?;
        if self.agent.is_none() {
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

                let decay = self.epsilon_decay;
                let action_selector = EpsilonGreedy::new(
                    self.epsilon,
                    self.rng.next_u64(),
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
                Some(DoubleDeepAgent::new(
                    action_selector,
                    mem_replay,
                    arch,
                    opt,
                    self.lr,
                    self.discount_factor,
                    Device::Cpu,
                ))
            })
        };

        let mut trainer = Trainer::new(env, py).unwrap();
        trainer.early_stop = Some(Box::new(move |reward| reward >= solve_with));

        let r: Result<TrainResults, OxiLearnErr> = trainer.train_by_steps(
            self.agent.as_mut().unwrap(),
            steps,
            update_freq,
            eval_at,
            eval_for,
            py,
        );
        py.allow_threads(|| {
            let rewards = r.unwrap().3;
            let reward_max = rewards
                .iter()
                .fold(rewards[0], |o, r| if *r > o { *r } else { o });
            Ok(reward_max)
        })
    }
}
