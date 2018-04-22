extern crate num_traits;
extern crate rand;

pub mod api;
pub mod array_ops;
pub mod categorical;
pub mod data;
pub mod iter_mean;
pub mod split;
pub mod vec2d;

mod testdata;


use std::marker::PhantomData;

use data::{SampleDescription, TrainingData};
use iter_mean::IterMean;
use split::{Split, SplitFinder};

/// A decision tree node.
///
/// Can be either a split node with a Splitter and two children, or a leaf
/// node with a LeafPredictor.
pub enum Node<T>
    where T: SampleDescription
{
    Invalid,  // placeholder used during tree construction
    Split{ theta: T::ThetaSplit, threshold: T::Feature, left: usize, right: usize},
    Leaf(T::ThetaLeaf),
}

/// A deterministic decision tree.
///
/// Splits are perfect - In contrast to probabilistic trees, a
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
                                               Feature=Sample::Feature> + ?Sized,
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

    pub fn print(&self)
        where Sample::ThetaLeaf: std::fmt::Debug
    {
        for node in &self.nodes {
            match node {
                &Node::Invalid => println!("INVALID NODE"),
                &Node::Leaf(ref l) => println!("Leaf Node: {:?}", l),
                &Node::Split{..} => println!("Split Node"),
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
    pub fn new(min_samples_split: usize, split_finder: SF) -> Self {
        DeterministicTreeBuilder {
            min_samples_split,
            split_finder,
            _p: PhantomData,
        }
    }

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


pub struct DeterministicForest<Sample>
    where Sample: SampleDescription
{
    estimators: Vec<DeterministicTree<Sample>>
}


impl<Sample> DeterministicForest<Sample>
    where Sample: SampleDescription
{
    // Making the predict function generic allows the user to pass in any sample that's compatible
    // with the tree's sample type
    pub fn predict<TestingSample>(&self, sample: &TestingSample) -> TestingSample::Prediction
        where TestingSample: SampleDescription<ThetaSplit=Sample::ThetaSplit,
                                               ThetaLeaf=Sample::ThetaLeaf,
                                               Feature=Sample::Feature> + ?Sized,
              TestingSample::Prediction: IterMean,
    {
        let iter = self.estimators
            .iter()
            .map(|tree| tree.predict(sample));
        TestingSample::Prediction::mean(iter)
    }

    pub fn print(&self)
        where Sample::ThetaLeaf: std::fmt::Debug
    {
        for (i, tree) in self.estimators.iter().enumerate() {
            println!("Tree {}:", i + 1);
            tree.print();
        }
    }
}


pub struct DeterministicForestBuilder<SF, Sample>
    where SF: SplitFinder,
          Sample: SampleDescription,
{
    n_estimators: usize,
    tree_builder: DeterministicTreeBuilder<SF, Sample>,
}

impl<SF, Sample> DeterministicForestBuilder<SF, Sample>
    where SF: SplitFinder,
          Sample: SampleDescription
{
    pub fn new(n_estimators: usize, tree_builder: DeterministicTreeBuilder<SF, Sample>) -> Self {
        DeterministicForestBuilder {
            n_estimators,
            tree_builder,
        }
    }

    pub fn fit<Training>(&self, data: &mut Training) -> DeterministicForest<Sample>
        where Training: ?Sized + TrainingData<Sample>
    {
        let mut estimators = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            estimators.push(self.tree_builder.fit(data));
        }

        DeterministicForest {
            estimators
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use split::BestRandomSplit;
    use testdata::Sample;

    #[test]
    fn tree() {
        let data: &mut [_] = &mut [
            Sample::new(&[0.0], 1.0),
            Sample::new(&[1.0], 2.0),
            Sample::new(&[2.0], 1.0),
            Sample::new(&[3.0], 2.0),
            Sample::new(&[4.0], 11.0),
            Sample::new(&[5.0], 12.0),
            Sample::new(&[6.0], 11.0),
            Sample::new(&[7.0], 12.0),
        ];

        let dtb = DeterministicTreeBuilder {
            _p: PhantomData,
            min_samples_split: 2,
            split_finder: BestRandomSplit::new(1),
        };

        let tree = dtb.fit(data);

        for sample in data {
            assert_eq!(tree.predict(sample), sample.y);
        }

    }
}
