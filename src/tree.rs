use crate::common::{
    log_sum_2_exp, normalize_hashmap_values, ClassProbabilities, Classes, ClassifierTarget,
    Direction, Observation,
};
use crate::node::{Node, Split};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Exp};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct MondrianTreeClassifier {
    step: f64,
    use_aggregation: bool,
    dirichlet: f64,
    split_pure: bool,
    seed: Option<usize>,
    rng: StdRng,
    iteration: usize,
    classes: Classes,
    root: usize,
    nodes: Vec<Node>, // To allow removing nodes might be best to store in hashmap
}

impl MondrianTreeClassifier {
    pub fn new(
        step: f64,
        use_aggregation: bool,
        dirichlet: f64,
        split_pure: bool,
        seed: Option<usize>,
    ) -> Self {
        MondrianTreeClassifier {
            step,
            use_aggregation,
            dirichlet,
            split_pure,
            seed,
            rng: StdRng::seed_from_u64(seed.unwrap_or(1) as u64),
            iteration: 0,
            classes: Classes::new(),
            root: 0,
            nodes: vec![Node::new_root()],
        }
    }

    fn is_initialised(&self) -> bool {
        self.classes.len() > 0
    }

    fn score(&self, y: &ClassifierTarget, node_idx: usize) -> f64 {
        self.nodes[node_idx].score(y, self.dirichlet, self.classes.len())
    }

    fn predict(&self, node_idx: usize) -> ClassProbabilities {
        self.nodes[node_idx].predict(self.dirichlet, &self.classes, self.classes.len())
    }

    fn loss(&self, y: &ClassifierTarget, node_idx: usize) -> f64 {
        self.nodes[node_idx].loss(y, self.dirichlet, self.classes.len())
    }

    fn update_weight(&mut self, node_idx: usize, y: &ClassifierTarget) {
        let node = &mut self.nodes[node_idx];
        node.update_weight(
            y,
            self.dirichlet,
            self.use_aggregation,
            self.step,
            self.classes.len(),
        );
    }

    fn update_count(&mut self, node_idx: usize, y: &ClassifierTarget) {
        let node = &mut self.nodes[node_idx];
        node.update_count(y);
    }

    fn update_depth(&mut self, node_idx: usize, depth: usize) {
        let node = &mut self.nodes[node_idx];
        node.update_depth(depth);
        // if it is not a leaf also update children
        let Some(split) = node.split.clone() else {
            return;
        };
        self.update_depth(split.left_child, depth + 1);
        self.update_depth(split.right_child, depth + 1);
    }

    fn update_downwards(
        &mut self,
        x: &Observation,
        y: &ClassifierTarget,
        node_idx: usize,
        do_weight_update: bool,
    ) {
        let node = &mut self.nodes[node_idx];
        let n_classes = self.classes.len();
        node.update_downwards(
            x,
            y,
            self.dirichlet,
            self.use_aggregation,
            self.step,
            do_weight_update,
            n_classes,
        )
    }

    fn exponential(&mut self, extensions_sum: f64) -> f64 {
        let exp = Exp::new(1.0 / extensions_sum).unwrap();
        exp.sample(&mut self.rng)
        // - (1.0 / extensions_sum) * self.uniform(0., 1.).ln() // todo: to make deterministic, this can be used
    }

    fn compute_split_time(
        &mut self,
        y: &ClassifierTarget,
        node_idx: usize,
        extensions_sum: f64,
    ) -> f64 {
        let node = &self.nodes[node_idx];
        if !self.split_pure && node.is_dirac(y) {
            return 0.;
        }

        if extensions_sum > 0. {
            let time = self.exponential(extensions_sum);
            // Splitting time of the node (if splitting occurs)
            let node = &self.nodes[node_idx];
            let split_time = time + node.time;
            // If the node is a leaf we must split it
            let Some(split) = &node.split else {
                return split_time;
            };

            // Otherwise we apply Mondrian process dark magic :)
            // 1. We get the creation time of the childs (left and right is the same)
            let child_time = self.nodes[split.left_child].time;
            // 2. We check if splitting time occurs before child creation time
            if split_time < child_time {
                return split_time;
            }
        }
        return 0.;
    }

    fn split_leaf(
        &mut self,
        node_idx: usize,
        split_time: f64,
        threshold: f64,
        feature: String,
        extend: Direction,
    ) -> usize {
        // We promote the leaf to a branch
        let left_idx = self.nodes.len();

        let node = &mut self.nodes[node_idx];
        let new_depth = node.depth + 1;
        let mut left = Node::new_leaf(left_idx, Some(node_idx), split_time, new_depth);

        let right_idx = left_idx + 1;
        let mut right = Node::new_leaf(right_idx, Some(node_idx), split_time, new_depth);

        match extend {
            Direction::Left => right.replant(&node),
            Direction::Right => left.replant(&node),
        };

        node.split(left_idx, right_idx, feature, threshold);
        self.nodes.push(left);
        self.nodes.push(right);
        node_idx
    }

    fn split_branch(
        &mut self,
        node_idx: usize,
        split_time: f64,
        threshold: f64,
        feature: String,
        extend: Direction,
    ) -> usize {
        let left_idx = self.nodes.len();
        let node = &mut self.nodes[node_idx];
        let new_depth = node.depth + 1;
        let node_split = node.split.clone().unwrap();

        let mut left = Node::new_leaf(left_idx, Some(node_idx), split_time, new_depth);

        let right_idx = left_idx + 1;
        let mut right = Node::new_leaf(right_idx, Some(node_idx), split_time, new_depth);

        let (new_parent, parent_idx) = match extend {
            Direction::Left => (&mut left, left_idx),
            Direction::Right => (&mut right, right_idx),
        };
        new_parent.replant(&node);
        node.split(left_idx, right_idx, feature, threshold);

        // Update the level of the modified nodes.
        let old_left = &self.nodes[node_split.left_child];
        self.update_depth(old_left.idx, new_depth + 1);
        let old_left = &mut self.nodes[node_split.left_child];
        old_left.parent = Some(parent_idx);

        let old_right = &self.nodes[node_split.right_child];
        self.update_depth(old_right.idx, new_depth + 1);
        let old_right = &mut self.nodes[node_split.right_child];
        old_right.parent = Some(parent_idx);

        new_parent.split(
            node_split.left_child,
            node_split.right_child,
            node_split.feature.clone(),
            node_split.threshold,
        );

        self.nodes.push(left);
        self.nodes.push(right);

        // Update split info.
        node_idx
    }

    fn split_branch_new(
        &mut self,
        node_idx: usize,
        split_time: f64,
        threshold: f64,
        feature: String,
        extend: Direction,
    ) -> usize {
        let left_idx = self.nodes.len();
        let node = &mut self.nodes[node_idx];
        let new_depth = node.depth + 1;
        let node_split = node.split.clone().unwrap();

        let mut left = Node::new_leaf(left_idx, Some(node_idx), split_time, new_depth);

        let right_idx = left_idx + 1;
        let mut right = Node::new_leaf(right_idx, Some(node_idx), split_time, new_depth);

        let (new_parent, parent_idx) = match extend {
            Direction::Right => (&mut left, left_idx),
            Direction::Left => (&mut right, right_idx),
        };
        new_parent.replant(&node);
        node.split(left_idx, right_idx, feature, threshold);

        // Update the level of the modified nodes.
        let old_left = &self.nodes[node_split.left_child];
        self.update_depth(old_left.idx, new_depth + 1);
        let old_left = &mut self.nodes[node_split.left_child];
        old_left.parent = Some(parent_idx);

        let old_right = &self.nodes[node_split.right_child];
        self.update_depth(old_right.idx, new_depth + 1);
        let old_right = &mut self.nodes[node_split.right_child];
        old_right.parent = Some(parent_idx);

        new_parent.split(
            node_split.left_child,
            node_split.right_child,
            node_split.feature.clone(),
            node_split.threshold,
        );

        self.nodes.push(left);
        self.nodes.push(right);

        // Update split info.
        node_idx
    }

    fn split_branch_new2(
        &mut self,
        node_idx: usize,
        split_time: f64,
        threshold: f64,
        feature: String,
        extend: Direction,
    ) -> usize {
        let left_idx = self.nodes.len();
        let right_idx = left_idx + 1;
        let node = &self.nodes[node_idx];
        let new_depth = node.depth + 1;
        let node_split = node.split.clone().unwrap();

        if extend == Direction::Right {
            let mut left = Node::new_leaf(left_idx, Some(node_idx), split_time, new_depth);

            let right = Node::new_leaf(right_idx, Some(node_idx), split_time, new_depth);

            left.replant(node);
            left.split(
                node_split.left_child,
                node_split.right_child,
                node_split.feature,
                node_split.threshold,
            );

            self.nodes[node_split.left_child].parent = Some(left.idx);
            self.nodes[node_split.right_child].parent = Some(left.idx);
            self.nodes.push(left);
            self.nodes.push(right);
        } else {
            let left = Node::new_leaf(left_idx, Some(node_idx), split_time, new_depth);

            let mut right = Node::new_leaf(right_idx, Some(node_idx), split_time, new_depth);

            right.replant(node);
            right.split(
                node_split.left_child,
                node_split.right_child,
                node_split.feature,
                node_split.threshold,
            );

            self.nodes[node_split.left_child].parent = Some(right.idx);
            self.nodes[node_split.right_child].parent = Some(right.idx);
            self.nodes.push(left);
            self.nodes.push(right);
        }

        // Update the level of the modified nodes
        self.update_depth(node_split.left_child, new_depth + 1);
        self.update_depth(node_split.right_child, new_depth + 1);
        // Update split info
        self.nodes[node_idx].split(left_idx, right_idx, feature, threshold);
        node_idx
    }

    fn split(
        &mut self,
        node_idx: usize,
        split_time: f64,
        threshold: f64,
        feature: String,
        extend: Direction,
    ) -> usize {
        let node = &self.nodes[node_idx];
        if node.is_leaf() {
            self.split_leaf(node_idx, split_time, threshold, feature, extend)
        } else {
            // The node is already a branch:
            // create a new branch above it and move the existing node one level down the tree
            self.split_branch_new2(node_idx, split_time, threshold, feature, extend)
        }
    }

    fn uniform(&mut self, min: f64, max: f64) -> f64 {
        self.rng.gen_range(min..max)
        // (min + max) / 2.0  // todo: to make deterministic
    }

    fn pick_random_key(&mut self, probabilities: &HashMap<String, f64>) -> Option<String> {
        let mut cumulative_prob = 0.0;
        let rand_num: f64 = self.rng.gen();

        for (key, &prob) in probabilities.iter() {
            cumulative_prob += prob;
            if rand_num < cumulative_prob {
                return Some(key.clone());
            }
        }
        None
    }

    fn most_common_path(&self, split: &Split) -> usize {
        let left_weight = self.nodes[split.left_child].weight;
        let right_weight = self.nodes[split.right_child].weight;

        if left_weight > right_weight {
            split.left_child
        } else {
            split.right_child
        }
    }

    fn go_downwards(&mut self, x: &Observation, y: &ClassifierTarget) -> usize {
        // We start at the root
        let mut current_idx = self.root;

        if self.iteration == 0 {
            // If it's the first iteration, we just put the current sample in the range of root
            self.update_downwards(x, y, current_idx, false);
            return current_idx;
        }
        loop {
            // Computing the extensions to get the intensities
            let (extensions_sum, extensions) = self.nodes[current_idx].range_extension(x);

            // If it's not the first iteration (otherwise the current node
            // is root with no range), we consider the possibility of a split
            let split_time = self.compute_split_time(y, current_idx, extensions_sum);
            if split_time > 0. {
                // We split the current node: because the current node is a
                // leaf, or because we add a new node along the path

                // We normalize the range extensions to get probabilities
                let intensities = normalize_hashmap_values(&extensions);

                // Sample the feature at random with a probability
                // proportional to the range extensions
                let feature = self.pick_random_key(&intensities).unwrap();
                let feature_value = *x.get(&feature).unwrap();

                let current_node = &self.nodes[current_idx];
                // Is it a right extension of the node ?
                let (range_min, range_max) = current_node.range(&feature);
                let (extend, threshold) = match feature_value > range_max {
                    true => (Direction::Right, self.uniform(range_max, feature_value)),
                    false => (Direction::Left, self.uniform(feature_value, range_min)),
                };

                // Split the current node.
                self.split(current_idx, split_time, threshold, feature, extend);

                // Update the current node
                self.update_downwards(x, y, current_idx, true);

                // Now, get the next node
                // we just did the split, so safe to unwrap
                let split = self.nodes[current_idx].split.as_ref().unwrap();
                current_idx = match extend {
                    Direction::Left => split.left_child,
                    Direction::Right => split.right_child,
                };

                // This is the leaf containing the sample point
                // (we've just split the current node with the data point)
                self.update_downwards(x, y, current_idx, false);
                return current_idx;
            } else {
                // There is no split, so we just update the node and go to the next one
                self.update_downwards(x, y, current_idx, true);
                let current_node = &self.nodes[current_idx];
                if let Some(new_direction) = current_node.direction(x) {
                    current_idx = current_node.get_child(new_direction).unwrap();
                } else {
                    let Some(split) = &current_node.split else {
                        // Arrived at the leaf, procedure is over
                        return current_idx;
                    };
                    current_idx = self.most_common_path(split);
                }
            }
        }
    }

    /// Update the weight of the node in the tree.
    pub fn update_weight_tree(&mut self, node_idx: usize) {
        let node = &self.nodes[node_idx];
        let new_weight: f64;
        if let Some(split) = &node.split {
            let left = &self.nodes[split.left_child];
            let right = &self.nodes[split.right_child];
            new_weight = log_sum_2_exp(node.weight, left.log_weight_tree + right.log_weight_tree)
        } else {
            new_weight = node.weight
        };
        let node = &mut self.nodes[node_idx];
        node.log_weight_tree = new_weight;
    }

    fn go_upwards(&mut self, leaf: usize) {
        if self.iteration == 0 {
            return;
        }

        let mut current_idx = leaf;
        loop {
            self.update_weight_tree(current_idx);
            let node = &self.nodes[current_idx];
            let Some(parent_idx) = node.parent else {
                return;
            };
            current_idx = parent_idx
        }
    }

    pub fn learn_one(&mut self, x: &Observation, y: &ClassifierTarget) {
        self.classes.insert(y.clone());

        // Learning step
        let leaf = self.go_downwards(x, y);
        if self.use_aggregation {
            self.go_upwards(leaf)
        }

        // Incrementing iteration
        self.iteration += 1;
    }

    fn find_leaf(&self, x: &Observation, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        if node.is_leaf() {
            return node.idx;
        }
        let next_idx = match node.next(x) {
            Some(idx) => idx,
            None => {
                let split = node.split.as_ref().unwrap();
                self.most_common_path(split)
            }
        };
        self.find_leaf(x, next_idx)
    }

    pub fn predict_proba_one(&self, x: &Observation) -> ClassProbabilities {
        if !self.is_initialised() {
            return HashMap::new();
        }

        let leaf_idx = self.find_leaf(x, self.root);

        if !self.use_aggregation {
            return self.predict(leaf_idx);
        }

        // Initialization of the scores to output to 0
        let mut scores = ClassProbabilities::new();
        for class in &self.classes {
            scores.insert(class.clone(), 0.);
        }
        let mut current_idx = leaf_idx;

        loop {
            // This test is useless ?
            let current_node = &self.nodes[current_idx];
            if current_node.is_leaf() {
                scores = self.predict(current_idx);
            } else {
                let weight = (current_node.weight - current_node.log_weight_tree).exp();
                // Get the predictions of the current node
                let predictions = self.predict(current_idx);
                for class in &self.classes {
                    let class_prediction = *predictions.get(&class).unwrap();
                    let class_score = *scores.get(&class).unwrap();

                    let new_score =
                        class_prediction * weight / 2. + (1. - weight / 2.) * class_score;
                    scores.insert(class.clone(), new_score);
                }
            }

            let Some(parent_idx) = current_node.parent else {
                // Arrived at the root
                break;
            };
            current_idx = parent_idx
        }

        // Normalize scores to mimic a probability distribution
        return normalize_hashmap_values(&scores);
    }

    pub fn print_tree(&self) {
        self.print_node(self.root, "", false);
    }

    fn print_node(&self, idx: usize, prefix: &str, is_last: bool) {
        let node = &self.nodes[idx];

        print!("{}", prefix);
        print!("{}", if is_last { "└── " } else { "├── " });

        if let Some(split) = &node.split {
            println!(
                "Node {}: time: {:?}, min: {:?}, max: {:?}, {:?}",
                idx, node.time, node.memory_range_min.inner, node.memory_range_max.inner, split
            );
            let new_prefix = if is_last { "    " } else { "│   " };
            self.print_node(
                split.left_child,
                &format!("{}{}", prefix, new_prefix),
                false,
            );
            self.print_node(
                split.right_child,
                &format!("{}{}", prefix, new_prefix),
                true,
            );
        } else {
            println!(
                "Node {}: time: {:?}, min: {:?}, max: {:?}, Leaf",
                idx, node.time, node.memory_range_min.inner, node.memory_range_max.inner,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_depth() {
        // PARAMETERS
        let mut tree = MondrianTreeClassifier::new(0.1, true, 0.5, false, Some(42));

        let new_depth = 10;
        let node_idx = 0;
        tree.update_depth(node_idx, new_depth);
        assert_eq!(new_depth, tree.nodes[node_idx].depth)
    }

    #[test]
    fn test_update_count() {
        // PARAMETERS
        let mut tree = MondrianTreeClassifier::new(0.1, true, 0.5, false, Some(42));

        let node_idx = 0;
        let target_false = ClassifierTarget::Bool(false);
        let count_false = 1;
        tree.update_count(node_idx, &target_false);
        assert_eq!(
            count_false,
            tree.nodes[node_idx].counts.count(&target_false)
        );

        let count_false = 2;
        tree.update_count(node_idx, &target_false);
        assert_eq!(
            count_false,
            tree.nodes[node_idx].counts.count(&target_false)
        );

        let target_true = ClassifierTarget::Bool(true);
        let count_true = 1;
        tree.update_count(node_idx, &target_true);
        assert_eq!(count_true, tree.nodes[node_idx].counts.count(&target_true));
        assert_eq!(
            count_false,
            tree.nodes[node_idx].counts.count(&target_false)
        );
    }
}
