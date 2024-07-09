use tch::{Device, Kind, Scalar, Tensor};
use torch_sys::IntList;

pub struct ExperienceStats {
    means: Tensor,
    msqs: Tensor,
    count: Tensor,
}

impl ExperienceStats {
    pub fn new(shape: impl IntList + Copy, device: Device) -> Self {
        Self {
            means: Tensor::zeros(shape, (Kind::Double, device)),
            msqs: Tensor::ones(shape, (Kind::Double, device)),
            count: Tensor::zeros(1, (Kind::Int64, device)),
        }
    }

    pub fn push(&mut self, value: &Tensor) {
        self.count += 1;
        let delta = value - &self.means;
        self.means += &delta / &self.count;
        let delta2 = value - &self.means;
        self.msqs += delta * delta2;
    }

    pub fn mean(&self) -> &Tensor {
        &self.means
    }

    pub fn var(&self) -> Tensor {
        &self.msqs / Scalar::float((self.count.int64_value(&[0]) - 1) as f64)
    }
}
