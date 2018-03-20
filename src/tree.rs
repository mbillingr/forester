use std::fmt;
use std::marker::PhantomData;

use super::DataSet;
use super::DeterministicSplitter;
use super::LeafPredictor;
use super::Sample;
use super::Side;
use super::SplitFitter;
use super::Splitter;

/// A decision tree node. Can be either a split node with a Splitter and two children, or a leaf
/// node with a LeafPredictor.
#[derive(Debug)]
enum Node<S: Splitter, L: LeafPredictor<S=<S::D as DataSet>::Item, D=S::D>>
{
    Invalid,  // placeholder used during tree construction
    Split{split: S, left: usize, right: usize},
    Leaf(L),
}

/// Generic decision tree.
#[derive(Debug)]
pub struct Tree<S: Splitter, P: LeafPredictor<S=<S::D as DataSet>::Item, D=S::D>>
{
    nodes: Vec<Node<S, P>>,
}

impl<S: Splitter, P: LeafPredictor<S=<S::D as DataSet>::Item, D=S::D>> Tree<S, P> {
    pub fn split_node(&mut self, n: usize, split: S) -> (usize, usize) {
        let left = self.nodes.len();
        let right = left + 1;
        self.nodes.push(Node::Invalid);
        self.nodes.push(Node::Invalid);
        self.nodes[n] = Node::Split{split, left, right};
        (left, right)
    }
}

impl<S: DeterministicSplitter, P: LeafPredictor<S=<S::D as DataSet>::Item, D=S::D>> Tree<S, P>
{
    /// Pass a sample `x` down the tree and predict output of final leaf.
    pub fn predict(&self, x: &<P::S as Sample>::X) -> <P::S as Sample>::Y {
        let mut n = 0;
        loop {
            match self.nodes[n] {
                Node::Split{ref split, left, right} => {
                    match split.split(x) {
                        Side::Left => n = left,
                        Side::Right => n = right,
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

struct TreeBuilder<S: SplitFitter, P: LeafPredictor<S=<S::D as DataSet>::Item, D=S::D>> {
    split_finder: S,
    min_samples_split: usize,
    _p: PhantomData<(S, P)>
}

impl<S: SplitFitter, P: LeafPredictor<S=<S::D as DataSet>::Item, D=S::D>> TreeBuilder<S, P>
    where S::Split: DeterministicSplitter
{
    pub fn fit(&self, data: &mut S::D) -> Tree<S::Split, P> {
        let mut tree = Tree {nodes: vec![Node::Invalid]};
        self.recursive_fit(&mut tree, data, 0);
        tree
    }

    fn recursive_fit(&self, tree: &mut Tree<S::Split, P>, data: &mut S::D, node: usize) {
        if data.n_samples() < self.min_samples_split {
            tree.nodes[node] = Node::Leaf(P::fit(data));
            return
        }
        let split = self.split_finder.find_split(data);
        match split {
            None => tree.nodes[node] = Node::Leaf(P::fit(data)),
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


#[cfg(test)]
mod tests {
    use super::*;
    use criteria::VarCriterion;
    use datasets::TupleSample;
    use features::ColumnSelect;
    use predictors::ConstMean;
    use splitters::BestRandomSplit;
    use splitters::ThresholdSplitter;

    #[test]
    fn predict() {
        type Sample = TupleSample<ColumnSelect, [i32;1], f64>;

        let tree: Tree<ThresholdSplitter<[Sample]>, ConstMean<_>> = Tree {
            nodes: vec!{
                Node::Split{split: ThresholdSplitter::new(0, 2), left: 1, right: 2},
                Node::Leaf(ConstMean::new(-1.0)),
                Node::Leaf(ConstMean::new(1.0))}
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
        type Data = [Sample];

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

        let tb: TreeBuilder<BestRandomSplit<ThresholdSplitter<_>, VarCriterion<_>>, ConstMean<Sample>> = TreeBuilder {
            split_finder: BestRandomSplit::new(1),
            min_samples_split: 2,
            _p: PhantomData,
        };

        let tree = tb.fit(&mut data);

        for sample in data {
            assert_eq!(tree.predict(&sample.get_x()), sample.get_y());
        }
    }
}
