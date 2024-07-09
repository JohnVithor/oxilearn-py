use tch::{Kind, Tensor};

#[derive(Debug)]
pub struct Categorical {
    probs: Tensor,
    logits: Tensor,
    batch_shape: Vec<i64>,
    num_events: i64,
}

impl Clone for Categorical {
    fn clone(&self) -> Self {
        Self {
            probs: self.probs.copy(),
            logits: self.logits.copy(),
            batch_shape: self.batch_shape.clone(),
            num_events: self.num_events,
        }
    }
}

impl Categorical {
    /// Creates a Categorical distribution from probabilities.
    pub fn from_probs(probs: Tensor) -> Self {
        let prob_sum = probs.sum_dim_intlist([-1].as_ref(), true, probs.kind());
        let probs = probs / prob_sum;

        let batch_shape: Vec<i64> = if probs.size().len() > 1 {
            probs.size().split_last().unwrap().1.to_vec()
        } else {
            vec![]
        };

        let num_events = *probs
            .size()
            .last()
            .expect("get last element of probs failed");

        Self {
            logits: probs_to_logits(&probs, false),
            probs,
            batch_shape,
            num_events,
        }
    }

    /// Creates a Categorical distribution from logits.
    pub fn from_logits(logits: Tensor) -> Self {
        let logsumexp = logits.logsumexp([-1], true);
        let logits = logits - logsumexp;

        let batch_shape: Vec<i64> = if logits.size().len() > 1 {
            logits.size().split_last().unwrap().1.to_vec()
        } else {
            vec![]
        };

        let num_events = *logits
            .size()
            .last()
            .expect("get last element of probs failed");

        Self {
            probs: logits_to_probs(&logits, false),
            logits,
            batch_shape,
            num_events,
        }
    }

    /// Returns the probabilities of the distribution.
    pub fn probs(&self) -> &Tensor {
        &self.probs
    }

    /// Returns the logits of the distribution.
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }

    /// Returns mean of the distribution.
    pub fn mean(&self) -> Tensor {
        tch::Tensor::full([], f64::NAN, (self.probs.kind(), self.probs.device()))
    }

    /// Returns variance of the distribution.
    pub fn variance(&self) -> Tensor {
        tch::Tensor::full([], f64::NAN, (self.probs.kind(), self.probs.device()))
    }

    pub fn entropy(&self) -> Tensor {
        let min_real = min(self.logits.kind()).unwrap();
        let logits = self.logits.clamp(min_real, f64::INFINITY);
        let p_log_p = logits * &self.probs;
        -p_log_p.sum_dim_intlist([-1].as_ref(), false, p_log_p.kind())
    }

    pub fn log_prob(&self, val: &Tensor) -> Tensor {
        let value = val.to_kind(tch::Kind::Int64).unsqueeze(-1);
        let value_log_pmf_vec = Tensor::broadcast_tensors(&[value, self.logits.copy()]);
        let value = &value_log_pmf_vec[0];
        let log_pmf = &value_log_pmf_vec[1];
        let tensor_index = Tensor::from_slice(&[0]).to_device(value.device());
        let value = value.index_select(-1, &tensor_index);
        log_pmf.gather(-1, &value, false).squeeze_dim(-1)
    }

    pub fn sample(&self, sample_shape: &[i64]) -> Tensor {
        let probs_2d = self.probs.reshape([-1, self.num_events]);
        let numel = sample_shape.iter().product();
        let x = probs_2d.multinomial(numel, true);
        let samples_2d = x.transpose(0, 1);
        let ext_shape = self.extended_shape(sample_shape);
        samples_2d.reshape(ext_shape)
    }

    pub fn extended_shape(&self, shape: &[i64]) -> Vec<i64> {
        [shape, self.batch_shape(), self.event_shape()].concat()
    }

    pub fn event_shape(&self) -> &[i64] {
        &[]
    }

    pub fn batch_shape(&self) -> &[i64] {
        &self.batch_shape
    }
}

pub fn min(kind: Kind) -> Option<f64> {
    // TODO: need refine
    Some(match kind {
        // Kind::Half => 6.103515625e-05,
        Kind::Float => f32::MIN as f64, //-3.4028234663852886e+38,
        Kind::Double => f64::MIN,       //-1.7976931348623157e+308,
        _ => return None,
    })
}

pub fn eps(kind: Kind) -> Option<f64> {
    Some(match kind {
        Kind::Half => 0.0009765625,
        Kind::Float => std::f32::EPSILON as _,
        Kind::Double => std::f64::EPSILON,
        _ => return None,
    })
}

fn clamp_probs(probs: &Tensor) -> Tensor {
    let eps = eps(probs.kind()).unwrap();
    probs.clamp(eps, 1.0 - eps)
}

pub fn probs_to_logits(probs: &Tensor, is_binary: bool) -> Tensor {
    let ps_clamped = clamp_probs(probs);
    if is_binary {
        ps_clamped.log() - (-ps_clamped).log1p()
    } else {
        ps_clamped.log()
    }
}

pub fn logits_to_probs(logits: &Tensor, is_binary: bool) -> Tensor {
    if is_binary {
        logits.sigmoid()
    } else {
        logits.softmax(-1, logits.kind())
    }
}
