use rand::{rngs::SmallRng, Rng, SeedableRng};
use tch::Tensor;

pub enum EpsilonUpdateStrategy {
    _AdaptativeEpsilon {
        min_epsilon: f32,
        max_epsilon: f32,
        min_reward: f32,
        max_reward: f32,
        eps_range: f32,
    },
    EpsilonDecreasing {
        final_epsilon: f32,
        epsilon_decay: Box<dyn Fn(f32) -> f32 + Send + Sync>,
    },
    None,
}

impl EpsilonUpdateStrategy {
    fn update(&mut self, current_epsilon: f32, epi_reward: f32) -> f32 {
        match self {
            EpsilonUpdateStrategy::_AdaptativeEpsilon {
                min_epsilon,
                max_epsilon,
                min_reward,
                max_reward,
                eps_range,
            } => {
                if epi_reward < *min_reward {
                    *max_epsilon
                } else {
                    let reward_range = *max_reward - *min_reward;
                    let min_update = *eps_range / reward_range;
                    let new_eps = (*max_reward - epi_reward) * min_update;
                    if new_eps < *min_epsilon {
                        *min_epsilon
                    } else {
                        new_eps
                    }
                }
            }
            EpsilonUpdateStrategy::EpsilonDecreasing {
                final_epsilon,
                epsilon_decay,
            } => {
                let new_epsilon: f32 = (epsilon_decay)(current_epsilon);

                if *final_epsilon > new_epsilon {
                    current_epsilon
                } else {
                    new_epsilon
                }
            }
            EpsilonUpdateStrategy::None => current_epsilon,
        }
    }
}

pub struct EpsilonGreedy {
    initial_epsilon: f32,
    epsilon: f32,
    rng: SmallRng,
    update_strategy: EpsilonUpdateStrategy,
}

impl Default for EpsilonGreedy {
    fn default() -> Self {
        Self::new(0.1, 42, EpsilonUpdateStrategy::None)
    }
}

impl EpsilonGreedy {
    pub fn new(epsilon: f32, seed: u64, update_strategy: EpsilonUpdateStrategy) -> Self {
        Self {
            initial_epsilon: epsilon,
            epsilon,
            rng: SmallRng::seed_from_u64(seed),
            update_strategy,
        }
    }

    pub fn should_explore(&mut self) -> bool {
        self.epsilon != 0.0 && self.rng.gen_range(0.0..1.0) <= self.epsilon
    }

    pub fn get_action(&mut self, values: &Tensor) -> usize {
        if self.should_explore() {
            self.rng.gen_range(0..values.size()[0] as usize)
        } else {
            let a: i32 = values.argmax(0, true).try_into().unwrap();
            a as usize
        }
    }

    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
    }

    pub fn update(&mut self, epi_reward: f32) {
        self.epsilon = self.update_strategy.update(self.epsilon, epi_reward)
    }
}
