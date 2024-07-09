use dqn::wrappers::DQN;
use env::cart_pole::CartPoleWrapper;
use pyo3::{
    create_exception, exceptions::PyException, pymodule, types::PyModule, Bound, PyResult, Python,
};

pub mod dqn;
pub mod env;
pub mod ppo;

#[derive(Debug, Clone)]
pub enum OxiLearnErr {
    MethodNotFound(String),
    ExpectedItemNotFound,
    DifferentTypeExpected,
    ExpectedDataMissing,
    SpaceNotSupported,
    EnvNotSupported,
}

/// A Python module implemented in Rust.
#[pymodule]
fn oxilearn(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", "0.0.1")?;
    m.add_class::<DQN>()?;
    m.add_class::<CartPoleWrapper>()?;
    create_exception!(m, OxiLearnErr, PyException);
    Ok(())
}
