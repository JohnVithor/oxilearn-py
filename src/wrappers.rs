use crate::dqn::DoubleDeepAgent;
use pyo3::{pyclass, pymethods};

#[pyclass]
pub struct DQNAgent {
    agent: DoubleDeepAgent,
}

#[pymethods]
impl DQNAgent {
    #[new]
    fn new() -> Self {
        Self {
            agent: DoubleDeepAgent::default(),
        }
    }

    pub fn train_by_steps2(
        &mut self,
        // env: todo!(),
        n_steps: u128,
        update_freq: u128,
        eval_at: u128,
        eval_for: u128,
        debug: bool,
    ) {
        // let mut training_reward: Vec<f32> = vec![];
        // let mut training_length: Vec<u128> = vec![];
        // let mut training_error: Vec<f32> = vec![];
        // let mut evaluation_reward: Vec<f32> = vec![];
        // let mut evaluation_length: Vec<f32> = vec![];

        // let mut n_episodes = 0;
        // let mut action_counter: u128 = 0;
        // let mut epi_reward: f32 = 0.0;
        // let mut curr_obs: Tensor = (self.obs_to_repr)(&self.train_env.reset());

        // for _ in 0..n_steps {
        //     let curr_action = agent.get_action(&curr_obs);

        //     let (next_obs, reward, terminated) = self
        //         .train_env
        //         .step((self.repr_to_action)(curr_action))
        //         .unwrap();
        //     let next_obs = (self.obs_to_repr)(&next_obs);

        //     agent.add_transition(
        //         &curr_obs,
        //         curr_action,
        //         reward,
        //         terminated,
        //         &next_obs,
        //         curr_action, // HERE
        //     );

        //     curr_obs = next_obs;

        //     if let Some(td) = agent.update() {
        //         training_error.push(td)
        //     }

        //     if terminated {
        //         if debug {
        //             println!("{}", self.train_env.render());
        //         }
        //         training_reward.push(epi_reward);
        //         if n_episodes % update_freq == 0 && agent.update_networks().is_err() {
        //             println!("copy error")
        //         }
        //         if n_episodes % eval_at == 0 {
        //             let (rewards, eval_lengths) = self.evaluate(agent, eval_for);
        //             let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
        //             let eval_lengths_avg = (eval_lengths.iter().map(|x| *x as f32).sum::<f32>())
        //                 / (eval_lengths.len() as f32);
        //             println!("Episode: {}, Avg Return: {:.3} ", n_episodes, reward_avg,);
        //             evaluation_reward.push(reward_avg);
        //             evaluation_length.push(eval_lengths_avg);
        //         }
        //         curr_obs = (self.obs_to_repr)(&self.train_env.reset());
        //         agent.action_selection_update(epi_reward);

        //         n_episodes += 1;
        //         epi_reward = 0.0;
        //         action_counter = 0;
        //     }
        //     training_length.push(action_counter);
        // }
    }
}
