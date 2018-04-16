extern crate rand;

pub mod api;
pub mod array_ops;
pub mod criteria;
pub mod datasets;
pub mod features;
pub mod ensemble;
pub mod get_item;
pub mod predictors;
pub mod random;
pub mod splitters;
pub mod traits;
pub mod d_tree;
//pub mod vec2d;

type Real = f64;

pub use traits::*;


use std::marker::PhantomData;
use rand::{thread_rng, Rng};
use array_ops::Partition;

pub struct Split<Theta, Threshold> {
    theta: Theta,
    threshold: Threshold,
}

/// Trait for a dataset that can be used for training a decision tree
pub trait TrainingData<Sample>
    where Sample: SampleDescription
{
    /// Return number of samples in the data set
    fn n_samples(&self) -> usize;

    /// Generate a new split feature (typically, this will be randomized)
    fn gen_split_feature(&self) -> Sample::ThetaSplit;

    /// Train a new leaf predictor
    fn train_leaf_predictor(&self) -> Sample::ThetaLeaf;

    /// Partition data set in-place according to a split
    fn partition_data(&mut self, split: &Split<Sample::ThetaSplit, Sample::Feature>) -> (&mut Self, &mut Self);

    /// Compute split criterion
    fn split_criterion(&self) -> f64;

    /// Return minimum and maximum value of a feature
    fn feature_bounds(&self, theta: &Sample::ThetaSplit) -> (Sample::Feature, Sample::Feature);
}

/// Trait that describes a Sample used with decision trees
pub trait SampleDescription {
    /// Type used to parametrize split features
    type ThetaSplit;

    /// Type used to parametrize leaf predictors
    type ThetaLeaf;

    /// Type of a split feature
    type Feature: PartialOrd + rand::distributions::range::SampleRange;

    /// Type of predicted values; this can be the same as `Self::Y` (e.g. regression) or something
    /// different (e.g. class probabilities).
    type Prediction;

    /// Compute the value of a leaf feature for a given sample
    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature;

    /// Compute the leaf prediction for a given sample
    fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction;
}

impl<'a, T> SampleDescription for (T, &'a[f64]) {
    type ThetaSplit = usize;
    type ThetaLeaf = f64;
    type Feature = f64;
    type Prediction = f64;

    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
        self.1[*theta]
    }

    fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction {
        *w
    }
}

impl<'a> TrainingData<(f64, &'a[f64])> for [(f64, &'a[f64])] {
    fn n_samples(&self) -> usize {
        self.len()
    }

    fn gen_split_feature(&self) -> usize {
        thread_rng().gen_range(0, self[0].1.len())
    }

    fn train_leaf_predictor(&self) -> f64 {
        self.iter().map(|&(y, _)| y).sum::<f64>() / self.len() as f64
    }

    /// Partition data set in-place according to a split
    fn partition_data(&mut self, split: &Split<usize, f64>) -> (&mut Self, &mut Self) {
        let i = self.partition(|sample| sample.sample_as_split_feature(&split.theta) <= split.threshold);
        self.split_at_mut(i)
    }

    /// Compute split criterion
    fn split_criterion(&self) -> f64 {
        let mean = self.iter().map(|&(y, _)| y).sum::<f64>() / self.len() as f64;
        self.iter().map(|&(y, _)| y - mean).map(|ym| ym * ym).sum::<f64>() / self.len() as f64
    }

    /// Return minimum and maximum value of a feature
    fn feature_bounds(&self, theta: &usize) -> (f64, f64) {
        self.iter()
            .map(|sample| sample.sample_as_split_feature(theta))
            .fold((std::f64::INFINITY, std::f64::NEG_INFINITY),
                         |(min, max), x| {
                             (if x < min {x} else {min},
                              if x > max {x} else {max})
        })
    }
}

/// Find split
pub trait SplitFinder
{
    fn find_split<Sample: SampleDescription,
                  Training: ?Sized + TrainingData<Sample>>(&self,
                                                           data: &mut Training)
        -> Option<Split<Sample::ThetaSplit, Sample::Feature>>;
}

/// A decision tree node. Can be either a split node with a Splitter and two children, or a leaf
/// node with a LeafPredictor.
pub enum Node<T>
    where T: SampleDescription
{
    Invalid,  // placeholder used during tree construction
    Split{ theta: T::ThetaSplit, threshold: T::Feature, left: usize, right: usize},
    Leaf(T::ThetaLeaf),
}

/// The usual type of decision tree. Splits are perfect - In contrast to probabilistic trees, a
/// sample deterministically goes down exactly one side of a split.
pub struct DeterministicTree<Sample>
    where Sample: SampleDescription
{
    nodes: Vec<Node<Sample>>
}

impl<Sample> DeterministicTree<Sample>
    where Sample: SampleDescription
{
    // Making the predict function generic allows the user to pass in any sample that's compatible
    // with the tree's sample type
    pub fn predict<TestingSample>(&self, sample: &TestingSample) -> TestingSample::Prediction
        where TestingSample: SampleDescription<ThetaSplit=Sample::ThetaSplit,
                                               ThetaLeaf=Sample::ThetaLeaf,
                                               Feature=Sample::Feature>,
    {
        let start = &self.nodes[0] as *const Node<Sample>;
        let mut node = &self.nodes[0] as *const Node<Sample>;
        unsafe {
            loop {
                match *node {
                    Node::Split { ref theta, ref threshold, left, right } => {
                        if &sample.sample_as_split_feature(theta) <= threshold {
                            node = start.offset(left as isize);
                        } else {
                            node = start.offset(right as isize);
                        }
                    }
                    Node::Leaf(ref l) => {
                        return sample.sample_predict(l)
                    }
                    Node::Invalid => panic!("Invalid node found. Tree may not be fully constructed.")
                }
            }
        }
    }
}


pub struct DeterministicTreeBuilder<SF, Sample>
    where SF: SplitFinder,
          Sample: SampleDescription,
{
    _p: PhantomData<Sample>,
    min_samples_split: usize,
    split_finder: SF,
}

impl<SF, Sample> DeterministicTreeBuilder<SF, Sample>
    where SF: SplitFinder,
          Sample: SampleDescription
{
    pub fn fit<Training>(&self, data: &mut Training) -> DeterministicTree<Sample>
        where Training: ?Sized + TrainingData<Sample>
    {
        let mut nodes = vec![Node::Invalid];
        self.recursive_fit(&mut nodes, data, 0);
        DeterministicTree {
            nodes
        }
    }

    fn recursive_fit<Training>(&self,
                               nodes: &mut Vec<Node<Sample>>,
                               data: &mut Training,
                               node: usize)
        where Training: ?Sized + TrainingData<Sample>
    {
        if data.n_samples() < self.min_samples_split {
            nodes[node] = Node::Leaf(data.train_leaf_predictor());
            return
        }
        let split = self.split_finder.find_split(data);
        match split {
            None => nodes[node] = Node::Leaf(data.train_leaf_predictor()),
            Some(split) => {
                let (left, right) = data.partition_data(&split);

                let (l, r) = Self::split_node(nodes, node, split);

                self.recursive_fit(nodes, left, l);
                self.recursive_fit(nodes, right, r);
            }
        }
    }

    fn split_node(nodes: &mut Vec<Node<Sample>>,
                      n: usize,
                      split: Split<Sample::ThetaSplit, Sample::Feature>)
        -> (usize, usize)
    {
        let left = nodes.len();
        let right = left + 1;
        nodes.push(Node::Invalid);
        nodes.push(Node::Invalid);
        nodes[n] = Node::Split{
            theta: split.theta,
            threshold:split.threshold,
            left,
            right};
        (left, right)
    }
}


pub struct BestRandomSplit {
    n_splits: usize,
}

impl SplitFinder for BestRandomSplit
{
    fn find_split<Sample, Training>(&self, data: &mut Training) -> Option<Split<Sample::ThetaSplit, Sample::Feature>>
        where Sample: SampleDescription,
              Training: ?Sized + TrainingData<Sample>
    {
        let n = data.n_samples() as f64;
        let mut best_criterion = data.split_criterion();
        let mut best_split = None;

        let mut rng = thread_rng();

        for _ in 0..self.n_splits {
            let theta = data.gen_split_feature();

            let (min, max) = data.feature_bounds(&theta);

            let threshold = rng.gen_range(min, max);

            let split = Split{theta, threshold};
            let (left, right) = data.partition_data(&split);

            let left_crit = left.split_criterion() * left.n_samples() as f64;
            let right_crit = right.split_criterion()* right.n_samples() as f64;
            let criterion = (left_crit + right_crit) / n;

            if criterion <= best_criterion {
                best_criterion = criterion;
                best_split = Some(split);
            }

            // stop early if we find a perfect split
            // TODO: tolerance rather than exact comparison
            if best_criterion == 0.0 {
                break
            }
        }

        best_split
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree() {
        let data: &mut [(f64, &[f64])] = &mut [
            (1.0, &[0.0]),
            (2.0, &[1.0]),
            (1.0, &[2.0]),
            (2.0, &[3.0]),
            (11.0, &[4.0]),
            (12.0, &[5.0]),
            (11.0, &[6.0]),
            (12.0, &[7.0]),
        ];

        let dtb = DeterministicTreeBuilder {
            _p: PhantomData,
            min_samples_split: 2,
            split_finder: BestRandomSplit {
                n_splits: 1,
            }
        };

        let tree = dtb.fit(data);

        for sample in data {
            assert_eq!(tree.predict(sample), sample.0);
        }

    }
}
