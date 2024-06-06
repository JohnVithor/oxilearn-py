use cart_pole::CartPoleWrapper;
use pyo3::{
    create_exception, exceptions::PyException, pymodule, types::PyModule, Bound, PyResult, Python,
};
use tch::{
    nn::{self, Module, VarStore},
    Device, Tensor,
};

pub mod cart_pole;
pub mod dqn;
pub mod env;
pub mod epsilon_greedy;
pub mod experience_buffer;
pub mod ppo;
pub mod trainer;
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

type PolicyGenerator = dyn Fn(Device) -> (Box<dyn Module>, VarStore);
type ActivationFunction = fn(&Tensor) -> Tensor;

pub fn generate_policy(
    net_arch: Vec<(i64, ActivationFunction)>,
    last_activation: ActivationFunction,
    input: i64,
    output: i64,
) -> Result<Box<PolicyGenerator>, OxiLearnErr> {
    Ok(Box::new(
        move |device: Device| -> (Box<dyn Module>, VarStore) {
            let iter = net_arch.clone().into_iter().enumerate();
            let mut previous = input;
            let mut mem_policy = VarStore::new(device);
            let mut policy_net = nn::seq();

            for (i, (neurons, activation)) in iter {
                policy_net = policy_net
                    .add(nn::linear(
                        &mem_policy.root() / format!("{}", i * 2),
                        previous,
                        neurons,
                        Default::default(),
                    ))
                    .add(nn::func(activation));
                previous = neurons;
            }
            policy_net = policy_net
                .add(nn::linear(
                    &mem_policy.root() / format!("{}", net_arch.len() * 2),
                    previous,
                    output,
                    Default::default(),
                ))
                .add(nn::func(last_activation));
            mem_policy.double();
            (Box::new(policy_net), mem_policy)
        },
    ))
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
