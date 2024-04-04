use std::collections::HashMap;
use crate::common::{ClassifierTarget, Counter, ClassProbabilities, Classes, Observation, FeatureRange, Direction};


#[derive(Clone, Debug)]
pub struct Split {
    pub left_child: usize,
    pub right_child: usize,
    pub feature: String,
    pub threshold: f64,
}

// impl Split {
//     pub fn update_depth(&mut self, depth: usize) {
//         self.right_child.borrow_mut().update_depth(depth);
//         self.left_child.borrow_mut().update_depth(depth);
//     }
// }


#[derive(Clone, Debug)]
pub struct Node {
    pub idx: usize,
    pub parent: Option<usize>,
    pub time: f64,
    pub depth: usize,
    pub n_samples: usize,
    pub counts: Counter,

    pub memory_range_min: FeatureRange,
    pub memory_range_max: FeatureRange,
    pub weight: f64,
    pub log_weight_tree: f64,

    pub split: Option<Split>,
}

impl Node {
    pub fn new_leaf(idx: usize, parent: Option<usize>, time: f64, depth: usize) -> Self {
        Self {
            idx,
            parent,
            time,
            depth,
            n_samples: 0,
            counts: Counter::new(),
            memory_range_min: FeatureRange::new(),
            memory_range_max: FeatureRange::new(),
            weight: 0.0,
            log_weight_tree: 0.0,
            split: None,
        }
    }

    pub fn new_root() -> Self {
        Self::new_leaf(0, None, 0., 0)
    }

    pub fn update_count(&mut self, y: &ClassifierTarget) {
        self.counts.update(y.clone());
    }

    pub fn update_depth(&mut self, depth: usize) {
        self.depth = depth;
    }

    pub fn update_weight(&mut self, y: &ClassifierTarget, dirichlet: f64, use_aggregation: bool,  step: f64,  n_classes: usize) {
        let loss = self.loss(y, dirichlet, n_classes);
        if use_aggregation {
            self.weight -= step * loss;
        }
    }

    pub fn predict(&self, dirichlet: f64, classes: &Classes, n_classes: usize) -> ClassProbabilities {
        let mut scores = ClassProbabilities::new();
        for class in classes {
            let class_score = self.score(class, dirichlet, n_classes);
            scores.insert(class.clone(), class_score);
        }
        scores
    }

    pub fn loss(&self, y: &ClassifierTarget, dirichlet: f64, n_classes: usize) -> f64 {
        let score = self.score(y, dirichlet, n_classes);
        -score.ln()
    }

    /// Compute the score of the node, uses Jeffreys prior with Dirichlet parameter for smoothing.
    pub fn score(&self, y: &ClassifierTarget, dirichlet: f64, n_classes: usize) -> f64 {
        let count = self.counts.count(y) as f64;
        (count + dirichlet) / (self.n_samples as f64 + dirichlet * n_classes as f64)
    }

    pub fn update_downwards(&mut self, x: &Observation, y: &ClassifierTarget, dirichlet: f64, use_aggregation: bool, step: f64, do_update_weight: bool, n_classes: usize) {
        // Updating the range of the feature values known by the node
        // If it is the first sample, we copy the features vector into the min and max range
        if self.n_samples == 0 {
            for (feature, value) in x {
                self.memory_range_min.insert(feature, value);
                self.memory_range_max.insert(feature, value);
            }
        } else {
            //Otherwise, we update the range
            for (feature, value) in x {
                if *value < self.memory_range_min.get(feature) {
                    self.memory_range_min.insert(feature, value);
                }
                if *value > self.memory_range_max.get(feature) {
                    self.memory_range_max.insert(feature, value);
                }
            }
        }

        // One more sample in the node
        self.n_samples += 1;

        if do_update_weight {
            self.update_weight(y, dirichlet, use_aggregation, step, n_classes)
        }
        self.update_count(y);
    }

    pub fn is_dirac(&self, y: &ClassifierTarget) -> bool {
        self.n_samples == self.counts.count(y)
    }

    pub fn is_leaf(&self) -> bool {
        match self.split {
            Some(_) => false,
            None => true,
        }
    }

    pub fn split(&mut self, left_child: usize, right_child: usize, feature: String, threshold: f64) {
        self.split = Some(
            Split {
                left_child,
                right_child,
                feature,
                threshold,
            }
        )
    }

    pub fn replant(&mut self, node: &Node) {
        self.weight = node.weight.clone();
        self.log_weight_tree = node.log_weight_tree.clone();
        // todo is this needed?
        // self.memory_range_min = node.memory_range_min.clone();
        // self.memory_range_max = node.memory_range_max.clone();
        // self.n_samples = node.n_samples;
        // self.counts = node.counts.clone();

    }

    pub fn range(&self, feature: &String) -> (f64, f64) {
        return (
            self.memory_range_min.get(feature),
            self.memory_range_max.get(feature),
        )
    }

    pub fn range_extension(&self, x: &Observation) -> (f64, HashMap<String, f64>) {
        let mut extensions= HashMap::new();
        let mut extensions_sum = 0.0;
        for (feature, &value) in x {
            let (feature_min, feature_max) = self.range(&feature);
            let diff: f64;
            if value < feature_min {
                diff = feature_min - value;
            } else if value > feature_max {
                diff = value - feature_max;
            } else {
                diff = 0.0;
            }
            extensions.insert(feature.clone(), diff);
            extensions_sum += diff;
        }
        (extensions_sum, extensions)
    }

    pub fn direction(&self, x: &Observation) -> Option<Direction> {
        let Some(split) = &self.split else {
            return None;
        };
        let feature_value = x.get(&split.feature);
        match feature_value {
            None => None,
            Some(&v) if v <= split.threshold => Some(Direction::Left),
            _ => Some(Direction::Right),
        }
    }

    pub fn get_child(&self, direction: Direction) -> Option<usize> {
        let Some(split) = &self.split else {
            return None;
        };
        match direction {
            Direction::Left => Some(split.left_child),
            Direction::Right => Some(split.right_child),
        }
    }

    pub fn next(&self, x: &Observation) -> Option<usize> {
        let Some(direction) = self.direction(x) else {
            return None;
        };
        self.get_child(direction)
    }
}