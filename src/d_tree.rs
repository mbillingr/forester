use std::marker::PhantomData;

use super::DataSet;
use super::DeterministicSplitter;
use super::LeafFitter;
use super::LeafPredictor;
use super::LearnerMut;
use super::Predictor;
use super::Sample;
use super::Side;
use super::SplitFitter;
use super::Splitter;

/// A decision tree node. Can be either a split node with a Splitter and two children, or a leaf
/// node with a LeafPredictor.
#[derive(Debug)]
pub enum Node<S: Sample, SP: Splitter<S>, L: LeafPredictor<S>>
{
    Invalid(PhantomData<S>),  // placeholder used during tree construction
    Split{split: SP, left: usize, right: usize},
    Leaf(L),
}

impl<S, SP, L> Node<S, SP, L>
    where S: Sample,
          SP: Splitter<S>,
          L: LeafPredictor<S>
{
    pub fn new() -> Self {
        Node::Invalid(PhantomData)
    }
}

/// Generic decision tree.
#[derive(Debug)]
pub struct DeterministicTree<S: Sample, SP: Splitter<S>, LP: LeafPredictor<S>>
{
    pub nodes: Vec<Node<S, SP, LP>>,  // Would rather make this private, but benchmarks need to have access
    _p: PhantomData<S>,
}

impl<S: Sample, SP: Splitter<S>, LP: LeafPredictor<S>> DeterministicTree<S, SP, LP> {
    pub fn new() -> Self {
        DeterministicTree {
            nodes: vec![Node::Invalid(PhantomData)],
            _p: PhantomData,
        }
    }

    pub fn split_node(&mut self, n: usize, split: SP) -> (usize, usize) {
        let left = self.nodes.len();
        let right = left + 1;
        self.nodes.push(Node::new());
        self.nodes.push(Node::new());
        self.nodes[n] = Node::Split{split, left, right};
        (left, right)
    }
}

/*
impl<S: ProbabilisticSplitter, P: LeafPredictor<S=<S::D as DataSet>::Item, D=S::D>> DeterministicTree<S, P>
{
    /// Pass a sample `x` down the tree on multiple paths and build final prediction based on the probability of each path.
    pub fn p_predict(&self, x: &<P::S as Sample>::X) -> <P::S as Sample>::Y {
        unimplemented!()
    }
}
*/

impl<S, SP, LP> Predictor<S::X> for DeterministicTree<S, SP, LP>
    where S: Sample,
          SP: DeterministicSplitter<S>,
          LP: LeafPredictor<S>
{
    type Output = LP::Output;

    /// Pass a sample `x` down the tree and predict output of final leaf.
    fn predict(&self, x: &S::X) -> LP::Output {
        let start = &self.nodes[0] as *const Node<S, SP, LP>;
        let mut node = &self.nodes[0] as *const Node<S, SP, LP>;
        unsafe {
            loop {
                match *node {
                    Node::Split { ref split, left, right } => {
                        match split.split(x) {
                            Side::Left => node = start.offset(left as isize),
                            Side::Right => node = start.offset(right as isize),
                        }
                    }
                    Node::Leaf(ref l) => {
                        return l.predict(x)
                    }
                    Node::Invalid(_) => panic!("Invalid node found. Tree may not be fully constructed.")
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct DeterministicTreeBuilder<S, SF, LP>
    where S: Sample,
          SF: SplitFitter<S>,
          LP: LeafFitter<S> + LeafPredictor<S>,
{
    split_finder: SF,
    min_samples_split: usize,
    _p: PhantomData<(S, LP)>
}

impl<S, SF, LP> DeterministicTreeBuilder<S, SF, LP>
    where S: Sample,
          SF: SplitFitter<S>,
          LP: LeafFitter<S> + LeafPredictor<S>,
{
    pub fn new(split_finder: SF, min_samples_split: usize) -> Self {
        DeterministicTreeBuilder {
            split_finder,
            min_samples_split,
            _p: PhantomData
        }
    }
}


impl<S, SF, LP> DeterministicTreeBuilder<S, SF, LP>
    where S: Sample,
          SF: SplitFitter<S>,
          LP: LeafFitter<S> + LeafPredictor<S>,
          SF::Split: DeterministicSplitter<S>
{
    fn recursive_fit(&self, tree: &mut DeterministicTree<S, SF::Split, LP>, data: &mut [S], node: usize) {
        if data.n_samples() < self.min_samples_split {
            tree.nodes[node] = Node::Leaf(LP::fit(data));
            return
        }
        let split = self.split_finder.find_split(data);
        match split {
            None => tree.nodes[node] = Node::Leaf(LP::fit(data)),
            Some(split) => {
                let i = data.partition_by_split(&split);
                let (left, right) = data.subsets(i);

                let (l, r) = tree.split_node(node, split);

                self.recursive_fit(tree, left, l);
                self.recursive_fit(tree, right, r);
            }
        }
    }
}

impl<S, SF, LP> Default for DeterministicTreeBuilder<S, SF, LP>
    where S: Sample,
          SF: SplitFitter<S>,
          LP: LeafFitter<S> + LeafPredictor<S>,
{
    fn default() -> Self {
        DeterministicTreeBuilder {
            split_finder: SF::default(),
            min_samples_split: 2,
            _p: PhantomData
        }
    }
}

impl<S, SF, LP> LearnerMut<S, DeterministicTree<S, SF::Split, LP>> for DeterministicTreeBuilder<S, SF, LP>
    where S: Sample,
          SF: SplitFitter<S>,
          LP: LeafFitter<S> + LeafPredictor<S>,
          SF::Split: DeterministicSplitter<S>,
{
    fn fit(&self, data: &mut [S]) -> DeterministicTree<S, SF::Split, LP> {
        let mut tree = DeterministicTree::new();
        self.recursive_fit(&mut tree, data, 0);
        tree
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use rand::thread_rng;

    use criteria::VarCriterion;
    use datasets::TupleSample;
    use features::ColumnSelect;
    use predictors::ConstMean;
    use splitters::BestRandomSplit;
    use splitters::ThresholdSplitter;

    #[test]
    fn predict() {
        type Sample = TupleSample<ColumnSelect, [i32;1], f64>;

        let tree: DeterministicTree<_, ThresholdSplitter<Sample>, ConstMean<_>> = DeterministicTree {
            nodes: vec!{
                Node::Split{split: ThresholdSplitter::new(0, 2), left: 1, right: 2},
                Node::Leaf(ConstMean::new(-1.0)),
                Node::Leaf(ConstMean::new(1.0))},
            _p: PhantomData
        };

        assert_eq!(tree.predict(&[-10]), -1.0);
        assert_eq!(tree.predict(&[1]), -1.0);
        assert_eq!(tree.predict(&[2]), -1.0);
        assert_eq!(tree.predict(&[3]), 1.0);
        assert_eq!(tree.predict(&[4]), 1.0);
        assert_eq!(tree.predict(&[10]), 1.0);
    }

    #[test]
    fn fit() {
        type Sample = TupleSample<ColumnSelect, [i32;1], f64>;

        let mut data: Vec<Sample> = vec![
            TupleSample::new([0], 1.0),
            TupleSample::new([1], 2.0),
            TupleSample::new([2], 1.0),
            TupleSample::new([3], 2.0),
            TupleSample::new([4], 11.0),
            TupleSample::new([5], 12.0),
            TupleSample::new([6], 11.0),
            TupleSample::new([7], 12.0),
            TupleSample::new([8], 5.0),
            TupleSample::new([9], 5.0),
            TupleSample::new([10], 5.0),
            TupleSample::new([11], 5.0),
        ];

        let tb: DeterministicTreeBuilder<_, BestRandomSplit<ThresholdSplitter<_>, VarCriterion<_>, _>, ConstMean<Sample>> = DeterministicTreeBuilder {
            split_finder: BestRandomSplit::new(1, thread_rng()),
            min_samples_split: 2,
            _p: PhantomData,
        };

        let tree = tb.fit(&mut data);

        for sample in data {
            assert_eq!(tree.predict(sample.get_x()), *sample.get_y());
        }
    }
}
