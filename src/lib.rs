use pyo3::{
    create_exception, exceptions::PyException, pymodule, types::PyModule, Bound, PyResult, Python,
};
use tch::{
    nn::{self, Module, VarStore},
    Device, Tensor,
};

mod dqn;
mod env;
mod epsilon_greedy;
mod experience_buffer;
mod trainer;
mod wrappers;

use wrappers::DQN;

#[derive(Debug, Clone)]
pub enum OxiLearnErr {
    MethodNotFound(String),
    ExpectedItemNotFound,
    DifferentTypeExpected,
    ExpectedDataMissing,
    SpaceNotSupported,
    EnvNotSupported,
}

type PolicyGenerator = dyn Fn(&str, Device) -> (Box<dyn Module>, VarStore);
type ActivationFunction = fn(&Tensor) -> Tensor;

fn generate_policy(
    net_arch: Vec<(i64, ActivationFunction)>,
    last_activation: ActivationFunction,
    input: i64,
    output: i64,
) -> Result<Box<PolicyGenerator>, OxiLearnErr> {
    Ok(Box::new(
        move |name: &str, device: Device| -> (Box<dyn Module>, VarStore) {
            let iter = net_arch.clone().into_iter().enumerate();
            let mut previous = input;
            let mem_policy = VarStore::new(device);
            let mut policy_net = nn::seq();

            for (i, (neurons, activation)) in iter {
                policy_net = policy_net
                    .add(nn::linear(
                        &mem_policy.root() / name / format!("{}", i * 2),
                        previous,
                        neurons,
                        Default::default(),
                    ))
                    .add_fn(activation);
                previous = neurons;
            }
            policy_net = policy_net
                .add(nn::linear(
                    &mem_policy.root() / name / format!("{}", net_arch.len() * 2),
                    previous,
                    output,
                    Default::default(),
                ))
                .add_fn(last_activation);
            (Box::new(policy_net), mem_policy)
        },
    ))
}

/// A Python module implemented in Rust.
#[pymodule]
fn oxilearn(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", "0.0.1")?;
    m.add_class::<DQN>()?;
    create_exception!(m, OxiLearnErr, PyException);
    Ok(())
}
