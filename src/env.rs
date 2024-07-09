use cart_pole::CartPole;
use pyo3::{Py, PyAny};

pub mod cart_pole;
pub mod pyenv;
pub mod space_info;

pub enum Env {
    Pyenv(Py<PyAny>),
    Native(CartPole),
}
