use pyo3::{pyclass, pymethods, Bound, PyAny, PyResult, Python};
use tch::{nn::Adam, Device};

use crate::env::PyEnv;

use oxilearn::{
    optimizer_enum::OptimizerEnum,
    ppo::{agent::PPOAgent, model::Policy, parameters::ParametersPPO},
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
    fn new(seed: i64) -> PyResult<Self> {
        Ok(Self { agent: None, seed })
    }

    #[pyo3(signature = ())]
    pub fn reset(&mut self) {
        self.agent = None;
    }

    #[pyo3(signature = (obs_size, n_action))]
    pub fn prepare(&mut self, obs_size: i64, n_action: i64) -> PyResult<()> {
        tch::manual_seed(self.seed);
        self.create_agent(obs_size, n_action)?;
        Ok(())
    }

    fn create_agent(&mut self, obs_size: i64, n_action: i64) -> PyResult<()> {
        // let obs_size = match environment.observation_space().unwrap() {
        //     SpaceInfo::Discrete(_) => Err(PyTypeError::new_err("ambiente inválido")),
        //     SpaceInfo::Continuous(s) => Ok(s.len()),
        // }? as i64;
        // let output = match environment.action_space().unwrap() {
        //     SpaceInfo::Discrete(n) => Ok(n),
        //     SpaceInfo::Continuous(_) => Err(PyTypeError::new_err("ambiente inválido")),
        // }? as i64;
        let device = Device::Cpu;
        let policy = Policy::new(obs_size, n_action, device);
        let optimizer = OptimizerEnum::Adam(Adam::default());
        let parameters = ParametersPPO::default();
        self.agent = Some(PPOAgent::new(
            policy, optimizer, obs_size, 100, parameters, device,
        ));
        Ok(())
    }

    #[pyo3(signature = (env, val_env, ))]
    #[allow(clippy::too_many_arguments)]
    pub fn train(&mut self, py: Python, env: Bound<PyAny>, val_env: Bound<PyAny>) -> PyResult<()> {
        let mut env = PyEnv::new(env)?;
        let mut val_env = PyEnv::new(val_env)?;
        py.allow_threads(|| {
            self.agent
                .as_mut()
                .unwrap()
                .learn(&mut env, &mut val_env, 100_000, self.seed, true)
        });
        Ok(())
    }

    #[pyo3(signature = (val_env, n_eval_episodes))]
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &mut self,
        val_env: Bound<PyAny>,
        n_eval_episodes: u32,
        py: Python,
    ) -> PyResult<Vec<f32>> {
        let mut val_env = PyEnv::new(val_env)?;

        Ok(py.allow_threads(|| -> Vec<f32> {
            self.agent
                .as_mut()
                .unwrap()
                .evaluate(&mut val_env, n_eval_episodes as usize)
        }))
    }
}
