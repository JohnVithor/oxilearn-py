use ndarray::{Array1, ArrayD};
use numpy::{PyArray, PyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use pyo3::{pyclass, pymethods, Bound, Py, PyAny, PyResult, Python};
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::env::SpaceInfo;

#[derive(Debug, Clone)]
pub struct CartPole {
    ready: bool,
    max_steps: i32,
    curr_step: i32,
    state: Array1<f32>,
    dist: Uniform<f32>,
    rng: SmallRng,
    pub action_space: SpaceInfo,
    pub observation_space: SpaceInfo,
}

fn initialize(dist: &Uniform<f32>, rng: &mut SmallRng) -> Array1<f32> {
    Array1::from_iter(dist.sample_iter(rng).take(4))
}

impl CartPole {
    pub const ACTIONS: [&'static str; 2] = ["PUSH TO THE LEFT", "PUSH TO THE RIGTH"];
    const GRAVITY: f32 = 9.8;
    const POLE_MASS: f32 = 0.1;
    const TOTAL_MASS: f32 = 1.1;
    const POLE_HALF_LENGTH: f32 = 0.5;
    const POLE_MASS_LENGTH: f32 = 0.05;
    const FORCE_MAG: f32 = 10.0;
    const TAU: f32 = 0.02;
    const THETA_THRESHOLD_RADIANS: f32 = std::f32::consts::PI / 15.0;
    const X_THRESHOLD: f32 = 2.4;

    pub fn new(max_steps: i32, seed: u64) -> Self {
        let mut env: CartPole = Self {
            ready: false,
            curr_step: 0,
            max_steps,
            state: Array1::<f32>::default(4),
            dist: Uniform::from(-0.05..0.05),
            rng: SmallRng::seed_from_u64(seed),
            action_space: SpaceInfo::Discrete(2),
            observation_space: SpaceInfo::Continuous(vec![
                (-4.8, 4.8),
                (f32::NEG_INFINITY, f32::INFINITY),
                (-0.418, 0.418),
                (f32::NEG_INFINITY, f32::INFINITY),
            ]),
        };
        env.state = initialize(&env.dist, &mut env.rng);
        env
    }

    pub fn reset(&mut self, seed: Option<u64>) -> ArrayD<f32> {
        if let Some(seed) = seed {
            self.rng = SmallRng::seed_from_u64(seed);
        }
        self.state = initialize(&self.dist, &mut self.rng);
        self.ready = true;
        self.curr_step = 0;
        self.state.clone().into_dyn()
    }

    pub fn step(&mut self, action: usize) -> Result<(ArrayD<f32>, f32, bool, bool), &'static str> {
        if !self.ready {
            return Err("Env not ready!");
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((self.state.clone().into_dyn(), 0.0, false, true));
        }
        self.curr_step += 1;

        let force = if action == 1 {
            Self::FORCE_MAG
        } else {
            -Self::FORCE_MAG
        };
        let cos_theta = self.state[2].cos();
        let sin_theta = self.state[2].sin();

        let temp = (force + Self::POLE_MASS_LENGTH * self.state[3] * self.state[3] * sin_theta)
            / Self::TOTAL_MASS;
        let thetaacc = (Self::GRAVITY * sin_theta - cos_theta * temp)
            / (Self::POLE_HALF_LENGTH
                * (4.0 / 3.0 - Self::POLE_MASS * cos_theta * cos_theta / Self::TOTAL_MASS));
        let xacc = temp - Self::POLE_MASS_LENGTH * thetaacc * cos_theta / Self::TOTAL_MASS;

        self.state[0] += Self::TAU * self.state[1];
        self.state[1] += Self::TAU * xacc;
        self.state[2] += Self::TAU * self.state[3];
        self.state[3] += Self::TAU * thetaacc;

        let terminated = self.state[0] < -Self::X_THRESHOLD
            || self.state[0] > Self::X_THRESHOLD
            || self.state[2] < -Self::THETA_THRESHOLD_RADIANS
            || self.state[2] > Self::THETA_THRESHOLD_RADIANS;
        Ok((self.state.clone().into_dyn(), 1.0, terminated, false))
    }
}

#[pyclass]
pub struct CartPoleWrapper {
    pub env: CartPole,
    pub observation_space: Py<PyAny>,
    pub action_space: Py<PyAny>,
}

#[pymethods]
impl CartPoleWrapper {
    #[new]
    #[pyo3(signature = (
        max_steps=500,
        seed=0,
    ))]
    pub fn new(max_steps: i32, seed: u64, py: Python) -> Self {
        let fun1 = PyModule::from_code_bound(
            py,
            "from gymnasium import spaces
def obs_space():
    return spaces.Discrete(2)
",
            "",
            "",
        )
        .unwrap()
        .getattr("obs_space")
        .unwrap();
        let fun2 = PyModule::from_code_bound(
            py,
            "from gymnasium import spaces
import numpy as np

def act_space():
    high = np.array(
        [
            4.8,
            np.finfo(np.float32).max,
            0.41887902047863906,
            np.finfo(np.float32).max,
        ],
        dtype=np.float32,
    )
    return spaces.Box(-high, high, dtype=np.float32)
",
            "",
            "",
        )
        .unwrap()
        .getattr("act_space")
        .unwrap();
        Self {
            env: CartPole::new(max_steps, seed),
            action_space: fun1.call0().unwrap().unbind(),
            observation_space: fun2.call0().unwrap().unbind(),
        }
    }
    #[getter]
    pub fn action_space(&self) -> &Py<PyAny> {
        &self.action_space
    }

    #[getter]
    pub fn observation_space(&self) -> &Py<PyAny> {
        &self.observation_space
    }

    #[pyo3(signature = (seed=None))]
    pub fn reset<'a>(
        &mut self,
        seed: Option<u64>,
        py: Python<'a>,
    ) -> PyResult<(Bound<'a, PyArrayDyn<f32>>, Py<PyDict>)> {
        let state = self.env.reset(seed);
        Ok((
            PyArray::from_array_bound(py, &state),
            PyDict::new_bound(py).unbind(),
        ))
    }

    #[pyo3(signature = (action))]
    pub fn step(
        &mut self,
        action: usize,
        py: Python,
    ) -> PyResult<(Py<PyArrayDyn<f32>>, f32, bool, bool, Py<PyDict>)> {
        let result = self.env.step(action);
        if let Ok((state, reward, terminated, done)) = result {
            return Ok((
                PyArray::from_array_bound(py, &state).unbind(),
                reward,
                terminated,
                done,
                PyDict::new_bound(py).unbind(),
            ));
        } else {
            Err(PyValueError::new_err("error"))
        }
    }
    pub fn close(&self) {}
    pub fn render(&self) {}
}
