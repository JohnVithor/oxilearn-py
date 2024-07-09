use tch::{
    nn::{Adam, AdamW, OptimizerConfig, RmsProp, Sgd},
    COptimizer, TchError,
};

#[derive(Debug, Copy, Clone)]
pub enum OptimizerEnum {
    Adam(Adam),
    Sgd(Sgd),
    RmsProp(RmsProp),
    AdamW(AdamW),
}

impl OptimizerConfig for OptimizerEnum {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        match self {
            OptimizerEnum::Adam(opt) => opt.build_copt(lr),
            OptimizerEnum::Sgd(opt) => opt.build_copt(lr),
            OptimizerEnum::RmsProp(opt) => opt.build_copt(lr),
            OptimizerEnum::AdamW(opt) => opt.build_copt(lr),
        }
    }
}
