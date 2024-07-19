use pyo3::{
    exceptions::{PyFileNotFoundError, PyTypeError, PyValueError},
    pyclass, pymethods, Bound, PyAny, PyResult, Python,
};
use rand::{rngs::SmallRng, RngCore, SeedableRng};
use tch::{
    nn::{Adam, AdamW, RmsProp, Sgd},
    Device, Kind, Tensor,
};

use crate::{
    dqn::optimizer_enum::OptimizerEnum,
    env::{PyEnv, SpaceInfo},
    OxiLearnErr,
};

use super::{
    agent::{OptimizationResults, PPOAgent},
    model::Policy,
};

#[pyclass]
pub struct PPO {
    agent: Option<PPOAgent>,
    seed: i64,
}

#[pymethods]
impl PPO {
    /// Create a new DQNAgent
    #[new]
    #[pyo3(signature = (seed))]
    fn new(py: Python, seed: i64) -> PyResult<Self> {
        tch::manual_seed(seed);
        Ok(Self { agent: None, seed })
    }

    #[pyo3(signature = ())]
    pub fn reset(&mut self) {
        self.agent = None;
    }

    #[pyo3(signature = (environment, val_environment))]
    pub fn prepare(
        &mut self,
        environment: Bound<PyAny>,
        val_environment: Bound<PyAny>,
    ) -> PyResult<()> {
        let val_env = PyEnv::new(val_environment)?;
        let environment = PyEnv::new(environment)?;
        self.create_agent(environment, val_env)?;
        Ok(())
    }

    fn create_agent(&mut self, environment: PyEnv, val_env: PyEnv) -> PyResult<()> {
        let input = match environment.observation_space().unwrap() {
            SpaceInfo::Discrete(_) => Err(PyTypeError::new_err("ambiente inválido")),
            SpaceInfo::Continuous(s) => Ok(s.len()),
        }? as i64;
        let output = match environment.action_space().unwrap() {
            SpaceInfo::Discrete(n) => Ok(n),
            SpaceInfo::Continuous(_) => Err(PyTypeError::new_err("ambiente inválido")),
        }? as i64;
        let device = Device::Cpu;
        let policy = Policy::new(input, output, device);
        let optimizer = OptimizerEnum::Adam(Adam::default());
        self.agent = Some(PPOAgent::new(
            policy,
            optimizer,
            environment,
            val_env,
            100,
            0.99,
            0.0005,
            0.95,
            64,
            16,
            4,
            0.2,
            true,
            true,
            0.01,
            0.5,
            0.5,
            Some(0.01),
            device,
        ));
        Ok(())
    }

    #[pyo3(signature = ())]
    #[allow(clippy::too_many_arguments)]
    pub fn train(&mut self, py: Python) {
        py.allow_threads(|| self.agent.as_mut().unwrap().learn(100_000, self.seed, true));
    }

    #[pyo3(signature = (n_eval_episodes))]
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(&mut self, n_eval_episodes: u32, py: Python) -> Vec<f32> {
        py.allow_threads(|| -> Vec<f32> {
            self.agent
                .as_mut()
                .unwrap()
                .evaluate(n_eval_episodes as usize)
        })
    }
}
