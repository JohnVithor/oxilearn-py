use pyo3::{
    create_exception,
    exceptions::{PyException, PyTypeError},
    pyfunction, pymodule,
    types::PyModule,
    wrap_pyfunction, Py, PyAny, PyResult, Python,
};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::time::Instant;
use tch::{
    nn::{self, Module, VarStore},
    Device, Kind, Tensor,
};

mod dqn;
mod env;
mod epsilon_greedy;
mod experience_buffer;
mod trainer;
mod wrappers;

use wrappers::DQNAgent;

use crate::{
    dqn::{DoubleDeepAgent, OptimizerEnum},
    env::{PyEnv, SpaceInfo},
    epsilon_greedy::{EpsilonGreedy, EpsilonUpdateStrategy},
    experience_buffer::RandomExperienceBuffer,
    trainer::{TrainResults, Trainer},
};

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

fn generate_policy(
    net_arch: Vec<(i64, ActivationFunction)>,
    last_activation: ActivationFunction,
    input: i64,
    output: i64,
) -> Result<Box<PolicyGenerator>, OxiLearnErr> {
    Ok(Box::new(
        move |device: Device| -> (Box<dyn Module>, VarStore) {
            let iter = net_arch.clone().into_iter().enumerate();
            let mut previous = input;
            let mem_policy = VarStore::new(device);
            let mut policy_net = nn::seq();

            for (i, (neurons, activation)) in iter {
                policy_net = policy_net
                    .add(nn::linear(
                        &mem_policy.root() / format!("layer {i}"),
                        previous,
                        neurons,
                        Default::default(),
                    ))
                    .add_fn(activation);
                previous = neurons;
            }
            policy_net = policy_net
                .add(nn::linear(
                    &mem_policy.root() / "output layer".to_string(),
                    previous,
                    output,
                    Default::default(),
                ))
                .add_fn(last_activation);
            (Box::new(policy_net), mem_policy)
        },
    ))
}

#[pyfunction]
pub fn test(env: Py<PyAny>) -> PyResult<f32> {
    let mut rng: StdRng = StdRng::seed_from_u64(4);

    tch::manual_seed(rng.next_u64() as i64);
    tch::maybe_init_cuda();
    const MEM_SIZE: usize = 5_000;
    const MIN_MEM_SIZE: usize = 1_000;
    const GAMMA: f32 = 0.99;
    const UPDATE_FREQ: u128 = 10;
    const LEARNING_RATE: f64 = 0.0005;
    const EPSILON_DECAY: f32 = 0.0005;
    const START_EPSILON: f32 = 1.0;
    let device: Device = Device::Cpu;

    let env = PyEnv::new(env)?;

    // let (s, r, d, t) = env.step(1).unwrap();
    // println!("{r}");
    // println!("{s}");
    // Ok(r)

    let mem_replay = RandomExperienceBuffer::new(MEM_SIZE, MIN_MEM_SIZE, rng.next_u64(), device);

    let epsilon_greedy = EpsilonGreedy::new(
        START_EPSILON,
        rng.next_u64(),
        EpsilonUpdateStrategy::EpsilonDecreasing {
            final_epsilon: 0.0,
            epsilon_decay: Box::new(move |a| a - EPSILON_DECAY),
        },
    );

    let input = match env.observation_space().unwrap() {
        SpaceInfo::Discrete(_) => Err(PyTypeError::new_err("ambiente inválido")),
        SpaceInfo::Continuous(s) => Ok(s.len()),
    }? as i64;
    let output = match env.action_space().unwrap() {
        SpaceInfo::Discrete(n) => Ok(n),
        SpaceInfo::Continuous(_) => Err(PyTypeError::new_err("ambiente inválido")),
    }? as i64;

    let net_arch = vec![(128, "relu".to_string()), (128, "softmax".to_string())];

    let mut info: Vec<(i64, ActivationFunction)> = Vec::new();
    for (n, activation) in net_arch {
        let activation = match activation.as_str() {
            "relu" => |xs: &Tensor| xs.relu(),
            "gelu" => |xs: &Tensor| xs.gelu("none"),
            "softmax" => |xs: &Tensor| xs.softmax(0, Kind::Float),
            _ => return Err(PyTypeError::new_err(activation)),
        };
        info.push((n, activation))
    }

    let arch = generate_policy(info, |xs: &Tensor| xs.shallow_clone(), input, output).unwrap();

    let mut agent = DoubleDeepAgent::new(
        epsilon_greedy,
        mem_replay,
        arch,
        OptimizerEnum::Adam(nn::Adam::default()),
        LEARNING_RATE,
        GAMMA,
        device,
    );

    let mut trainer = Trainer::new(env).unwrap();
    trainer.early_stop = Some(Box::new(|reward| reward >= 500.0));

    let start = Instant::now();

    let r: Result<TrainResults, OxiLearnErr> =
        trainer.train_by_steps(&mut agent, 200_000, UPDATE_FREQ, 50, 10);
    let elapsed = start.elapsed();
    println!("Elapsed time: {elapsed:?}");
    let rewards = r.unwrap().3;
    let reward_max = rewards
        .iter()
        .fold(rewards[0], |o, r| if *r > o { *r } else { o });
    Ok(reward_max)
}

/// A Python module implemented in Rust.
#[pymodule]
fn oxilearn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.0.1")?;
    m.add_class::<DQNAgent>()?;
    m.add_function(wrap_pyfunction!(test, m)?).unwrap();
    create_exception!(m, OxiLearnErr, PyException);
    Ok(())
}
