use crate::common::ClassifierTarget;

pub struct Accuracy {
    matched_num: usize,
    total_num: usize,
}

impl Accuracy {
    pub fn new() -> Self {
        Self {
            matched_num: 0,
            total_num: 0,
        }
    }
}

// implement for trait ClassificationMetric
impl Accuracy {
    pub fn update(&mut self, y_true: &ClassifierTarget, y_pred: &ClassifierTarget) {
        self.total_num += 1;
        if y_pred == y_true {
            self.matched_num += 1;
        }
    }

    pub fn get(&self) -> f64 {
        self.matched_num as f64 / self.total_num as f64
    }
}
