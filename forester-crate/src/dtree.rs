//! Deterministic Tree module
//!
//! A deterministic tree is a tree where a sample is passed down on exactly one side of each split,
//! depending on feature and threshold. (This stands in contrast to probabilistic trees, where
//! samples are passed down both sides with certain probabilities.)

use std::fmt;
use std::marker::PhantomData;

use data::{SampleDescription, TrainingData};
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
pub struct DeterministicTree<Sample>
    where Sample: SampleDescription
{
    nodes: Vec<Node<Sample>>
}

impl<Sample: SampleDescription> fmt::Debug for DeterministicTree<Sample>
    where Sample::ThetaLeaf: fmt::Debug,
          Sample::ThetaSplit: fmt::Debug,
          Sample::Feature: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tree:")?;
        let mut prefix = vec![];
        self.recursive_fmt(0, &mut prefix, false, f)
    }
}

impl<Sample: SampleDescription> DeterministicTree<Sample>
    where Sample::ThetaLeaf: fmt::Debug,
          Sample::ThetaSplit: fmt::Debug,
          Sample::Feature: fmt::Debug,
{
    fn recursive_fmt(&self, n: usize, prefix: &mut Vec<&str>, bottom: bool, f: &mut fmt::Formatter) -> fmt::Result {
        let node = &self.nodes[n];

        writeln!(f)?;
        for p in prefix.iter() {
            write!(f, "{}", p)?;
        }

        if bottom {
            prefix.pop();
            prefix.push("    ");
        }

        match *node {
            Node::Invalid => write!(f, " *** Invalid ***")?,
            Node::Leaf(ref l) => write!(f, " {:?}", l)?,
            Node::Split{ref theta, ref threshold, left, right} => {
                write!(f, "({:?}) <= {:?}", theta, threshold)?;
                if let Some(&" +--") = prefix.last() {
                    prefix.pop();
                    prefix.push(" |  ");
                }
                prefix.push(" +--");
                self.recursive_fmt(left, prefix, false, f)?;
                prefix.pop();
                prefix.push(" +--");
                self.recursive_fmt(right, prefix, true, f)?;
                prefix.pop();
            }
        }

        Ok(())
    }
}

impl<Sample> DeterministicTree<Sample>
    where Sample: SampleDescription
{
    #[cfg(test)]
    pub(crate) fn new_with_nodes(nodes: Vec<Node<Sample>>) -> Self {
        DeterministicTree {
            nodes
        }
    }

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
}

/// Fit a `DeterministicTree` to `TrainingData`.
pub struct DeterministicTreeBuilder<SF, Sample>
    where SF: SplitFinder,
          Sample: SampleDescription,
{
    _p: PhantomData<Sample>,
    min_samples_split: usize,
    max_depth: Option<usize>,
    split_finder: SF,
}

impl<SF, Sample> DeterministicTreeBuilder<SF, Sample>
    where SF: SplitFinder,
          Sample: SampleDescription
{
    pub fn new(min_samples_split: usize, max_depth: Option<usize>, split_finder: SF) -> Self {
        DeterministicTreeBuilder {
            min_samples_split,
            split_finder,
            max_depth,
            _p: PhantomData,
        }
    }

    pub fn fit<Training>(&self, data: &mut Training) -> DeterministicTree<Sample>
        where Training: ?Sized + TrainingData<Sample>
    {
        let mut nodes = vec![Node::Invalid];
        self.recursive_fit(&mut nodes, data, 0, 0);
        DeterministicTree {
            nodes
        }
    }

    fn recursive_fit<Training>(&self,
                               nodes: &mut Vec<Node<Sample>>,
                               data: &mut Training,
                               node: usize,
                               depth: usize)
        where Training: ?Sized + TrainingData<Sample>
    {
        if let Some(md) = self.max_depth {
            if depth >= md {
                nodes[node] = Node::Leaf(data.train_leaf_predictor());
                return
            }
        }

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

                self.recursive_fit(nodes, left, l, depth + 1);
                self.recursive_fit(nodes, right, r, depth + 1);
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
            max_depth: None,
            split_finder: BestRandomSplit::new(1),
        };

        let tree = dtb.fit(data);

        for sample in data {
            assert_eq!(tree.predict(sample), sample.y);
        }

    }

    #[test]
    fn fmt() {
        let tree: DeterministicTree<Sample<_, _>> = DeterministicTree {
            nodes: vec![
                Node::Split { theta: 1, threshold: 2.3, left: 1, right: 2},
                Node::Leaf(4.5),
                Node::Invalid,
            ]
        };

        let formatted = format!("{:?}", tree);

        assert_eq!(formatted, "Tree:\n(1) <= 2.3\n +-- 4.5\n +-- *** Invalid ***");
    }
}
