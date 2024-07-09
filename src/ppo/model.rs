use tch::nn::LinearConfig;
use tch::nn::ModuleT;
use tch::nn::SequentialT;
use tch::{nn, Tensor};

use super::categorical::Categorical;

#[derive(Debug)]
pub struct Policy {
    critic: SequentialT,
    actor: SequentialT,
}

impl Policy {
    pub fn new(vs: &nn::Path, obs_shape: i64, num_actions: i64) -> Self {
        let config = LinearConfig {
            ws_init: tch::nn::Init::Orthogonal {
                gain: 2.0_f64.sqrt(),
            },
            bs_init: Some(tch::nn::Init::Const(0.0)),
            bias: true,
        };

        let config2 = LinearConfig {
            ws_init: tch::nn::Init::Orthogonal { gain: 1.0 },
            bs_init: Some(tch::nn::Init::Const(0.0)),
            bias: true,
        };

        let config3 = LinearConfig {
            ws_init: tch::nn::Init::Orthogonal { gain: 0.01 },
            bs_init: Some(tch::nn::Init::Const(0.0)),
            bias: true,
        };

        let critic = nn::seq_t()
            .add(nn::linear(vs, obs_shape, 64, config))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(vs, 64, 64, config))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(vs, 64, 1, config2));

        let actor = nn::seq_t()
            .add(nn::linear(vs, obs_shape, 64, config))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(vs, 64, 64, config))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(vs, 64, num_actions, config3));

        Policy { critic, actor }
    }

    pub fn get_value(&self, x: &Tensor) -> Tensor {
        self.critic.forward_t(x, true)
    }

    pub fn get_action_and_value(
        &self,
        x: &Tensor,
        action: Option<Tensor>,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let logits = self.actor.forward_t(x, true);
        let probs = Categorical::from_logits(logits);

        let action = match action {
            Some(a) => a,
            None => probs.sample(&[]),
        };

        let log_prob = probs.log_prob(&action);
        let entropy = probs.entropy();
        let value = self.critic.forward_t(x, true);

        (action, log_prob, entropy, value)
    }

    pub fn get_best_action(&self, x: &Tensor) -> Tensor {
        let logits = self.actor.forward_t(x, true);
        logits.argmax(-1, true)
    }
}
