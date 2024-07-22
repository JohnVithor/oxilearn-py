use dqn::DQN;
use ppo::PPO;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

pub mod dqn;
pub mod env;
pub mod ppo;

/// A Python module implemented in Rust.
#[pymodule]
fn oxilearnpy(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", "0.0.1")?;
    m.add_class::<DQN>()?;
    m.add_class::<PPO>()?;
    Ok(())
}
