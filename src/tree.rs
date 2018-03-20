
use super::DataSet;
use super::DeterministicSplitter;
use super::LeafPredictor;
use super::Sample;
use super::Side;
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


#[cfg(test)]
mod tests {
    use super::*;
    use datasets::TupleSample;
    use features::ColumnSelect;
    use predictors::ConstMean;
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
}
