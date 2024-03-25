use numpy::{ndarray::Array1, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyTypeError, types::PyTuple, Py, PyAny, PyResult, Python};
use tch::Tensor;

use crate::OxiLearnErr;

pub struct PyEnv {
    env: Py<PyAny>,
}

pub enum SpaceInfo {
    Discrete(usize),
    Continuous(Vec<(f32, f32)>),
}

impl SpaceInfo {
    pub fn is_discrete(&self) -> bool {
        match self {
            SpaceInfo::Discrete(_) => true,
            SpaceInfo::Continuous(_) => false,
        }
    }
}

impl PyEnv {
    pub fn new(env: Py<PyAny>) -> PyResult<Self> {
        Python::with_gil(|py: Python<'_>| -> PyResult<Self> {
            let a = env.as_ref(py);
            if !a.hasattr("reset").unwrap() {
                return Err(PyTypeError::new_err(
                    "Object hasn't 'reset' method!".to_string(),
                ));
            }
            if !a.hasattr("step").unwrap() {
                return Err(PyTypeError::new_err(
                    "Object hasn't 'step' method!".to_string(),
                ));
            }
            if !a.hasattr("action_space").unwrap() {
                return Err(PyTypeError::new_err(
                    "Object hasn't 'action_space' attribute!".to_string(),
                ));
            }
            Ok(Self { env })
        })
    }

    fn extract_state(&self, resulting_tuple: &PyTuple) -> Result<Tensor, OxiLearnErr> {
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

    fn extract_reward(&self, resulting_tuple: &PyTuple) -> Result<f32, OxiLearnErr> {
        let Ok(start_obs) = resulting_tuple.get_item(1) else {
            return Err(OxiLearnErr::ExpectedItemNotFound);
        };
        let Ok(reward) = start_obs.extract::<f32>() else {
            return Err(OxiLearnErr::DifferentTypeExpected);
        };
        Ok(reward)
    }

    fn extract_done(&self, resulting_tuple: &PyTuple) -> Result<bool, OxiLearnErr> {
        let Ok(start_obs) = resulting_tuple.get_item(2) else {
            return Err(OxiLearnErr::ExpectedItemNotFound);
        };
        let Ok(done) = start_obs.extract::<bool>() else {
            return Err(OxiLearnErr::DifferentTypeExpected);
        };
        Ok(done)
    }

    fn extract_truncated(&self, resulting_tuple: &PyTuple) -> Result<bool, OxiLearnErr> {
        let Ok(start_obs) = resulting_tuple.get_item(3) else {
            return Err(OxiLearnErr::ExpectedItemNotFound);
        };
        let Ok(truncated) = start_obs.extract::<bool>() else {
            return Err(OxiLearnErr::DifferentTypeExpected);
        };
        Ok(truncated)
    }

    fn extract_space(
        &self,
        attribute: Py<PyAny>,
        py: Python<'_>,
    ) -> Result<SpaceInfo, OxiLearnErr> {
        let space = attribute.as_ref(py).get_type().getattr("__name__").unwrap();
        let name: String = space.extract().unwrap();
        if name.eq("Discrete") {
            let size = attribute.as_ref(py).getattr("n").unwrap();
            Ok(SpaceInfo::Discrete(size.extract().unwrap()))
        } else if name.eq("Box") {
            let high = attribute.as_ref(py).getattr("high").unwrap();
            let high = high.extract::<PyReadonlyArrayDyn<f32>>().unwrap();
            let high = high.as_array();
            let low = attribute.as_ref(py).getattr("low").unwrap();
            let low = low.extract::<PyReadonlyArrayDyn<f32>>().unwrap();
            let low = low.as_array();

            Ok(SpaceInfo::Continuous(
                low.iter().zip(high).map(|(l, h)| (*l, *h)).collect(),
            ))
        } else {
            Err(OxiLearnErr::SpaceNotSupported)
        }
    }

    pub fn reset(&mut self) -> Result<Tensor, OxiLearnErr> {
        Python::with_gil(|py| -> Result<Tensor, OxiLearnErr> {
            // let kwargs = [("seed", 0)].into_py_dict(py);
            let Ok(call_result) = self.env.call_method(py, "reset", (), None) else {
                return Err(OxiLearnErr::MethodNotFound("reset".to_string()));
            };
            let Ok(resulting_tuple) = call_result.downcast::<PyTuple>(py) else {
                return Err(OxiLearnErr::DifferentTypeExpected);
            };
            self.extract_state(resulting_tuple)
        })
    }

    pub fn step(&mut self, action: usize) -> Result<(Tensor, f32, bool, bool), OxiLearnErr> {
        Python::with_gil(|py| -> Result<(Tensor, f32, bool, bool), OxiLearnErr> {
            let Ok(call_result) =
                self.env
                    .call_method(py, "step", PyTuple::new(py, [action]), None)
            else {
                return Err(OxiLearnErr::MethodNotFound("step".to_string()));
            };
            let Ok(resulting_tuple) = call_result.downcast::<PyTuple>(py) else {
                return Err(OxiLearnErr::DifferentTypeExpected);
            };
            let state = self.extract_state(resulting_tuple)?;
            let reward = self.extract_reward(resulting_tuple)?;
            let done = self.extract_done(resulting_tuple)?;
            let truncated = self.extract_truncated(resulting_tuple)?;
            Ok((state, reward, done, truncated))
        })
    }

    pub fn render(&self) -> String {
        todo!()
    }

    pub fn observation_space(&self) -> Result<SpaceInfo, OxiLearnErr> {
        Python::with_gil(|py| -> Result<SpaceInfo, OxiLearnErr> {
            let attribute = self.env.getattr(py, "observation_space").unwrap();
            self.extract_space(attribute, py)
        })
    }

    pub fn action_space(&self) -> Result<SpaceInfo, OxiLearnErr> {
        Python::with_gil(|py| -> Result<SpaceInfo, OxiLearnErr> {
            let attribute = self.env.getattr(py, "action_space").unwrap();
            self.extract_space(attribute, py)
        })
    }
}
