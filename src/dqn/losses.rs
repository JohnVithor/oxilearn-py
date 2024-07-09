use tch::Tensor;

pub fn mae(values: &Tensor, expected_values: &Tensor) -> Tensor {
    values.l1_loss(expected_values, tch::Reduction::Mean)
}

pub fn mse(values: &Tensor, expected_values: &Tensor) -> Tensor {
    values.mse_loss(expected_values, tch::Reduction::Mean)
}

pub fn rmse(values: &Tensor, expected_values: &Tensor) -> Tensor {
    values
        .mse_loss(expected_values, tch::Reduction::Mean)
        .sqrt()
}

pub fn huber(values: &Tensor, expected_values: &Tensor) -> Tensor {
    values.huber_loss(expected_values, tch::Reduction::Mean, 1.35)
}

pub fn smooth_l1(values: &Tensor, expected_values: &Tensor) -> Tensor {
    values.smooth_l1_loss(expected_values, tch::Reduction::Mean, 1.0 / 9.0)
}
