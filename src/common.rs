use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use rand::Rng;


#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClassifierTarget {
    Bool(bool),
    Int(i32),
    String(String),
}


pub type Observation = HashMap<String, f64>;

#[derive(Debug, Clone)]
pub struct FeatureRange {
    pub inner: HashMap<String, f64>
}


impl FeatureRange {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new()
        }
    }

    pub fn get(&self, feature: &String) -> f64 {
        *self.inner.get(feature).unwrap_or(&0.)
    }

    pub fn insert(&mut self, feature: &String, value: &f64) {
        self.inner.insert(feature.clone(), *value);
    }
}

pub type ClassProbabilities = HashMap<ClassifierTarget, f64>;

pub type Classes = HashSet<ClassifierTarget>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Debug, Clone)]
pub struct Counter {
    counter: HashMap<ClassifierTarget, usize>,
}

impl Counter {
    pub fn new() -> Self {
        Counter {
            counter: HashMap::new(),
        }
    }

    pub fn update(&mut self, item: ClassifierTarget) {
        *self.counter.entry(item).or_insert(0) += 1;
    }

    pub fn count(&self, item: &ClassifierTarget) -> usize {
        *self.counter.get(item).unwrap_or(&0)
    }

    // fn most_common(&self) -> Vec<(&ClassifierTarget, &usize)> {
    //     let mut items: Vec<(&ClassifierTarget, &usize)> = self.counter.iter().collect();
    //     items.sort_by(|a, b| b.1.cmp(a.1));
    //     items
    // }
}

pub fn normalize_hashmap_values<K: Eq + Hash + Clone>(map: &HashMap<K, f64>) -> HashMap<K, f64> {
    let sum: f64 = map.values().cloned().sum();
    let mut normalized_map = HashMap::new();
    // Normalize each value in the HashMap
    for (key, &value) in map {
        normalized_map.insert(key.clone(), value / sum);
    }
    normalized_map
}


/// Computation of log( (e^a + e^b) / 2) in an overflow-proof way
pub fn log_sum_2_exp(a: f64, b: f64) -> f64 {
    if a > b {
        a + ((1. + (b - a).exp()) / 2.).ln()
    } else {
        b + ((1. + (a - b).exp()) / 2.).ln()
    }
}
