use numpy::{ndarray::Array1, PyReadonlyArrayDyn};
use oxilearn::env::{Env, SpaceInfo};
use pyo3::{
    exceptions::PyTypeError,
    pyclass,
    types::{IntoPyDict, PyAnyMethods, PyTuple},
    Bound, Py, PyAny, PyResult, Python,
};
use tch::Tensor;

use oxilearn::OxiLearnErr;

#[pyclass]
#[derive(Clone)]
pub struct PyEnv {
    env: Py<PyAny>,
    pub reward_threshold: Option<f32>,
}

impl PyEnv {
    pub fn new(env: Bound<PyAny>) -> PyResult<Self> {
        if !env.hasattr("reset").unwrap() {
            return Err(PyTypeError::new_err(
                "Object hasn't 'reset' method!".to_string(),
            ));
        }
        if !env.hasattr("step").unwrap() {
            return Err(PyTypeError::new_err(
                "Object hasn't 'step' method!".to_string(),
            ));
        }
        if !env.hasattr("action_space").unwrap() {
            return Err(PyTypeError::new_err(
                "Object hasn't 'action_space' attribute!".to_string(),
            ));
        }

        let reward_threshold = if env.hasattr("spec").unwrap() {
            let a = env.getattr("spec").unwrap();
            if a.hasattr("reward_threshold").unwrap() {
                let a = a.getattr("reward_threshold").unwrap();
                let value = a.extract::<f32>().unwrap();
                Some(value)
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self {
            env: env.unbind(),
            reward_threshold,
        })
    }
}

impl Env for PyEnv {
    fn reset(&mut self, seed: Option<i64>) -> Result<Tensor, OxiLearnErr> {
        Python::with_gil(|py| {
            let kwargs = seed.map(|seed| [("seed", seed)].into_py_dict_bound(py));
            let kwargs = kwargs.as_ref();
            let Ok(call_result) = self.env.call_method_bound(py, "reset", (), kwargs) else {
                return Err(OxiLearnErr::MethodNotFound("reset".to_string()));
            };
            let Ok(resulting_tuple) = call_result.downcast_bound::<PyTuple>(py) else {
                return Err(OxiLearnErr::DifferentTypeExpected);
            };
            extract_state(resulting_tuple)
        })
    }

    fn step(&mut self, action: usize) -> Result<(Tensor, f32, bool, bool), OxiLearnErr> {
        Python::with_gil(|py| {
            let Ok(call_result) =
                self.env
                    .call_method_bound(py, "step", PyTuple::new_bound(py, [action]), None)
            else {
                return Err(OxiLearnErr::MethodNotFound("step".to_string()));
            };
            let Ok(resulting_tuple) = call_result.downcast_bound::<PyTuple>(py) else {
                return Err(OxiLearnErr::DifferentTypeExpected);
            };
            let state = extract_state(resulting_tuple)?;
            let reward = extract_reward(resulting_tuple)?;
            let done = extract_done(resulting_tuple)?;
            let truncated = extract_truncated(resulting_tuple)?;
            Ok((state, reward, done, truncated))
        })
    }

    fn observation_space(&self) -> Result<SpaceInfo, OxiLearnErr> {
        Python::with_gil(|py| {
            let attribute = self.env.getattr(py, "observation_space").unwrap();
            extract_space(attribute.bind(py))
        })
    }

    fn action_space(&self) -> Result<SpaceInfo, OxiLearnErr> {
        Python::with_gil(|py| {
            let attribute = self.env.getattr(py, "action_space").unwrap();
            extract_space(attribute.bind(py))
        })
    }

    fn reward_threshold(&self) -> Option<f32> {
        self.reward_threshold
    }
}

fn extract_state(resulting_tuple: &Bound<PyTuple>) -> Result<Tensor, OxiLearnErr> {
    let Ok(start_obs) = resulting_tuple.get_item(0) else {
        return Err(OxiLearnErr::ExpectedItemNotFound);
    };
    match start_obs.extract::<PyReadonlyArrayDyn<f32>>() {
        Ok(arr_data) => {
            let binding = arr_data.as_array();
            let Some(slice) = binding.as_slice() else {
                return Err(OxiLearnErr::ExpectedDataMissing);
            };
            Ok(Array1::from_vec(Vec::from(slice))
                .into_dyn()
                .try_into()
                .unwrap())
        }
        Err(_) => {
            let Ok(elem) = start_obs.extract::<usize>() else {
                return Err(OxiLearnErr::DifferentTypeExpected);
            };
            Ok(Array1::from_elem(1, elem as f32)
                .into_dyn()
                .try_into()
                .unwrap())
        }
    }
}

fn extract_reward(resulting_tuple: &Bound<PyTuple>) -> Result<f32, OxiLearnErr> {
    let Ok(start_obs) = resulting_tuple.get_item(1) else {
        return Err(OxiLearnErr::ExpectedItemNotFound);
    };
    let Ok(reward) = start_obs.extract::<f32>() else {
        return Err(OxiLearnErr::DifferentTypeExpected);
    };
    Ok(reward)
}

fn extract_done(resulting_tuple: &Bound<PyTuple>) -> Result<bool, OxiLearnErr> {
    let Ok(start_obs) = resulting_tuple.get_item(2) else {
        return Err(OxiLearnErr::ExpectedItemNotFound);
    };
    let Ok(done) = start_obs.extract::<bool>() else {
        return Err(OxiLearnErr::DifferentTypeExpected);
    };
    Ok(done)
}

fn extract_truncated(resulting_tuple: &Bound<PyTuple>) -> Result<bool, OxiLearnErr> {
    let Ok(start_obs) = resulting_tuple.get_item(3) else {
        return Err(OxiLearnErr::ExpectedItemNotFound);
    };
    let Ok(truncated) = start_obs.extract::<bool>() else {
        return Err(OxiLearnErr::DifferentTypeExpected);
    };
    Ok(truncated)
}

fn extract_space(attribute: &Bound<PyAny>) -> Result<SpaceInfo, OxiLearnErr> {
    let space = attribute.get_type().getattr("__name__").unwrap();
    let name: String = space.extract().unwrap();
    if name.eq("Discrete") {
        let size = attribute.getattr("n").unwrap();
        Ok(SpaceInfo::Discrete(size.extract().unwrap()))
    } else if name.eq("Box") {
        let high = attribute.getattr("high").unwrap();
        let high = high.extract::<PyReadonlyArrayDyn<f32>>().unwrap();
        let high = high.as_array();
        let low = attribute.getattr("low").unwrap();
        let low = low.extract::<PyReadonlyArrayDyn<f32>>().unwrap();
        let low = low.as_array();

        Ok(SpaceInfo::Continuous(
            low.iter().zip(high).map(|(l, h)| (*l, *h)).collect(),
        ))
    } else {
        Err(OxiLearnErr::SpaceNotSupported)
    }
}
