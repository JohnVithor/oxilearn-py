use oxilearn::{
    cart_pole::CartPole,
    dqn::DoubleDeepAgent,
    epsilon_greedy::EpsilonGreedy,
    experience_buffer::RandomExperienceBuffer,
    generate_policy,
    trainer::{TrainResults, Trainer},
    OxiLearnErr,
};
use std::env;
use tch::{Device, Tensor};
fn main() {
    let args: Vec<String> = env::args().collect();
    let seed = args[1].parse::<u64>().unwrap();
    tch::manual_seed(seed as i64);
    tch::maybe_init_cuda();

    let device = Device::cuda_if_available();
    let train_env = CartPole::new(500, seed);
    let eval_env = CartPole::new(500, seed + 1);

    let update_strategy =
        oxilearn::epsilon_greedy::EpsilonUpdateStrategy::EpsilonLinearTrainingDecreasing {
            start: 1.0,
            end: 0.05,
            end_fraction: 0.1,
        };
    let action_selector = EpsilonGreedy::new(1.0, seed + 2, update_strategy);

    let mem_replay = RandomExperienceBuffer::new(10_000, 4, 1_000, seed + 3, false, device);
    let policy = generate_policy(
        vec![
            (256, |xs: &Tensor| xs.relu()),
            (256, |xs: &Tensor| xs.relu()),
        ],
        |xs: &Tensor| xs.shallow_clone(),
        4,
        2,
    )
    .unwrap();
    let opt = oxilearn::dqn::OptimizerEnum::Adam(tch::nn::Adam::default());
    let loss_fn = |pred: &Tensor, target: &Tensor| pred.mse_loss(target, tch::Reduction::Mean);

    let mut model = DoubleDeepAgent::new(
        action_selector,
        mem_replay,
        policy,
        opt,
        loss_fn,
        0.0001,
        0.99,
        10.0,
        device,
    );

    let mut trainer = Trainer::new(train_env, eval_env);
    trainer.early_stop = Some(Box::new(move |reward| reward >= 475.0));

    let training_results: Result<TrainResults, OxiLearnErr> =
        trainer.train_by_steps(&mut model, 50_000, 128, 256, 64, 10, 1000, 10, 1);

    let training_steps = training_results.unwrap().1.iter().sum::<u32>();

    let evaluation_results = trainer.evaluate(&mut model, 10);
    let rewards = evaluation_results.unwrap().0;
    let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
    let variance = rewards
        .iter()
        .map(|value| {
            let diff = reward_avg - *value;
            diff * diff
        })
        .sum::<f32>()
        / rewards.len() as f32;
    let std = variance.sqrt();
    println!("rust,{seed},{training_steps},{reward_avg},{std}")
}
