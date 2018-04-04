use std::marker::PhantomData;

use super::DataSet;
use super::DeterministicSplitter;
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
pub enum Node<S: Splitter, L: LeafPredictor<S::S>>
{
    Invalid,  // placeholder used during tree construction
    Split{split: S, left: usize, right: usize},
    Leaf(L),
}

/// Generic decision tree.
#[derive(Debug)]
pub struct DeterministicTree<S: Sample, SP: Splitter<S=S>, LP: LeafPredictor<S>>
{
    pub nodes: Vec<Node<SP, LP>>,  // Would rather make this private, but benchmarks need to have access
    _p: PhantomData<S>,
}

impl<S: Sample, SP: Splitter<S=S>, LP: LeafPredictor<S>> DeterministicTree<S, SP, LP> {
    pub fn split_node(&mut self, n: usize, split: SP) -> (usize, usize) {
        let left = self.nodes.len();
        let right = left + 1;
        self.nodes.push(Node::Invalid);
        self.nodes.push(Node::Invalid);
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

impl<S: Sample, SP: DeterministicSplitter<S=S>, LP: LeafPredictor<S>> Predictor<S::X> for DeterministicTree<S, SP, LP>
{
    type Output = LP::Output;

    /// Pass a sample `x` down the tree and predict output of final leaf.
    fn predict(&self, x: &S::X) -> LP::Output {
        let start = &self.nodes[0] as *const Node<SP, LP>;
        let mut node = &self.nodes[0] as *const Node<SP, LP>;
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
                    Node::Invalid => panic!("Invalid node found. Tree may not be fully constructed.")
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct DeterministicTreeBuilder<S: SplitFitter, P: LeafPredictor<S::S>> {
    split_finder: S,
    min_samples_split: usize,
    _p: PhantomData<(S, P)>
}

impl<S: SplitFitter, P: LeafPredictor<S::S>> DeterministicTreeBuilder<S, P> {
    pub fn new(split_finder: S, min_samples_split: usize) -> DeterministicTreeBuilder<S, P> {
        DeterministicTreeBuilder {
            split_finder,
            min_samples_split,
            _p: PhantomData
        }
    }
}


impl<SF: SplitFitter, LP: LeafPredictor<SF::S>> DeterministicTreeBuilder<SF, LP>
    where SF::Split: DeterministicSplitter
{
    fn recursive_fit(&self, tree: &mut DeterministicTree<SF::S, SF::Split, LP>, data: &mut [SF::S], node: usize) {
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

impl<S: SplitFitter, P: LeafPredictor<S::S>> Default for DeterministicTreeBuilder<S, P> {
    fn default() -> DeterministicTreeBuilder<S, P> {
        DeterministicTreeBuilder {
            split_finder: S::default(),
            min_samples_split: 2,
            _p: PhantomData
        }
    }
}

impl<SF: SplitFitter, LP: LeafPredictor<SF::S>> LearnerMut<SF::S, DeterministicTree<SF::S, SF::Split, LP>> for DeterministicTreeBuilder<SF, LP>
    where SF::Split: DeterministicSplitter
{
    fn fit(&self, data: &mut [SF::S]) -> DeterministicTree<SF::S, SF::Split, LP> {
        let mut tree = DeterministicTree {
            nodes: vec![Node::Invalid],
            _p: PhantomData,
        };
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

        let tb: DeterministicTreeBuilder<BestRandomSplit<ThresholdSplitter<_>, VarCriterion<_>, _>, ConstMean<Sample>> = DeterministicTreeBuilder {
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
