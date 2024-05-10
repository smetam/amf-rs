use crate::common::{ClassProbabilities, Classes, ClassifierTarget, Observation};
use crate::tree::MondrianTreeClassifier;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashSet;

#[derive(Debug)]
pub struct AMFClassifier {
    step: f64,
    use_aggregation: bool,
    dirichlet: f64,
    split_pure: bool,
    seed: Option<usize>,
    classes: HashSet<ClassifierTarget>,
    trees: Vec<MondrianTreeClassifier>,
    rng: StdRng,
}

impl Default for AMFClassifier {
    fn default() -> Self {
        Self {
            step: 1.0,
            use_aggregation: true,
            dirichlet: 0.5,
            split_pure: false,
            seed: None,
            classes: Classes::new(),
            trees: Vec::with_capacity(10),
            rng: StdRng::seed_from_u64(1),
        }
    }
}

impl AMFClassifier {
    pub fn new(
        n_estimators: usize,
        step: f64,
        use_aggregation: bool,
        dirichlet: f64,
        split_pure: bool,
        seed: Option<usize>,
    ) -> Self {
        let mut trees = Vec::with_capacity(n_estimators);
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(1) as u64);
        for _ in 0..n_estimators {
            let random_seed = rng.gen_range(0..=9999999);
            let tree = MondrianTreeClassifier::new(
                step,
                use_aggregation,
                dirichlet,
                split_pure,
                Some(random_seed), // todo: trees should have different seed
            );
            trees.push(tree)
        }
        Self {
            step,
            use_aggregation,
            dirichlet,
            split_pure,
            seed,
            rng,
            classes: Classes::new(),
            trees,
        }
    }

    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    pub fn learn_one(&mut self, x: &Observation, y: &ClassifierTarget) {
        // Updating the previously seen classes with the new sample.
        self.classes.insert(y.clone());

        // Fit all the trees using the new sample.
        for tree in &mut self.trees {
            tree.learn_one(x, y)
        }
    }

    pub fn predict_proba_one(&self, x: &Observation) -> ClassProbabilities {
        // Initialize the scores.
        let mut scores = ClassProbabilities::new();

        // Checking that the model has been trained at least once.
        // Otherwise, return empty predictions.
        if self.classes.len() == 0 {
            return scores;
        }

        let n_estimators = self.n_estimators() as f64;
        // Compute the prediction for each amf and average.
        for tree in &self.trees {
            let predictions = tree.predict_proba_one(x);
            for class in &self.classes {
                let new_score = *predictions.get(&class).unwrap_or(&0.);
                let new_score = new_score / n_estimators;
                *scores.entry(class.clone()).or_insert(0.) += new_score;
            }
        }
        scores
    }
}
