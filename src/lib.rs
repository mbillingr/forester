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

struct Split<Theta, Threshold> {
    theta: Theta,
    threshold: Threshold,
}

/// Trait for a dataset that can be used for training a decision tree
trait TrainingData<Sample>:
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
    fn split_criterion(&self);
}

/// Trait that describes a Sample used with decision trees
trait SampleDescription {
    /// Type used to parametrize split features
    type ThetaSplit;

    /// Type used to parametrize leaf predictors
    type ThetaLeaf;

    /// Type of a split feature
    type Feature: PartialOrd;

    /// Type of predicted values; this can be the same as `Self::Y` (e.g. regression) or something
    /// different (e.g. class probabilities).
    type Prediction;

    /// Compute the value of a leaf feature for a given sample
    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature;

    /// Compute the leaf prediction for a given sample
    fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction;
}

/// Find split
trait SplitFinder
{
    fn find_split<Sample: SampleDescription,
                  Training: TrainingData<Sample>>(&self,
                                                  data: &Training)
        -> Option<Split<Sample::ThetaSplit, Sample::Feature>>;
}

/// A decision tree node. Can be either a split node with a Splitter and two children, or a leaf
/// node with a LeafPredictor.
enum Node<T>
    where T: SampleDescription
{
    Invalid,  // placeholder used during tree construction
    Split{ theta: T::ThetaSplit, threshold: T::Feature, left: usize, right: usize},
    Leaf(T::ThetaLeaf),
}

/// The usual type of decision tree. Splits are perfect - In contrast to probabilistic trees, a
/// sample deterministically goes down exactly one side of a split.
struct DeterministicTree<Sample>
    where Sample: SampleDescription
{
    nodes: Vec<Node<Sample>>
}

impl<Sample> DeterministicTree<Sample>
    where Sample: SampleDescription
{
    // Making the predict function generic allows the user to pass in any sample that's compatible
    // with the tree's sample type
    fn predict<TestingSample>(&self, sample: &TestingSample) -> TestingSample::Prediction
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


struct DeterministicTreeBuilder<SF, Sample>
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
    fn fit<Training>(&self, data: &mut Training) -> DeterministicTree<Sample>
        where Training: TrainingData<Sample>
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
        where Training: TrainingData<Sample>
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

    pub fn split_node(nodes: &mut Vec<Node<Sample>>,
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


#[cfg(test)]
mod tests {
}
