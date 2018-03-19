/*use super::DeterministicSplitter;
use super::LeafPredictor;
use super::Side;
use super::Splitter;

/// A decision tree node. Can be either a split node with a Splitter and two children, or a leaf
/// node with a LeafPredictor.
#[derive(Debug)]
enum Node<S: Splitter, L: LeafPredictor<X=S::X>>
{
    Invalid,  // placeholder used during tree construction
    Split{split: S, left: usize, right: usize},
    Leaf(L),
}

/// Generic decision tree.
#[derive(Debug)]
pub struct Tree<S: Splitter, P: LeafPredictor<X=S::X>>
{
    nodes: Vec<Node<S, P>>,
}

impl<S: Splitter, P: LeafPredictor<X=S::X>> Tree<S, P> {
    pub fn split_node(&mut self, n: usize, split: S) -> (usize, usize) {
        let left = self.nodes.len();
        let right = left + 1;
        self.nodes.push(Node::Invalid);
        self.nodes.push(Node::Invalid);
        self.nodes[n] = Node::Split{split, left, right};
        (left, right)
    }
}

impl<S: DeterministicSplitter, P: LeafPredictor<X=S::X>> Tree<S, P>
where <P::Y as OutcomeVariable>::Item: Sized
{
    /// Pass a sample `x` down the tree and predict output of final leaf.
    pub fn predict(&self, x: &<S::X as FeatureSet>::Item) -> <P::Y as OutcomeVariable>::Item {
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
    use vec2d::Vec2D;
    use features::ColumnFeature;
    use splitters::ThresholdSplitter;
    use predictors::ConstMean;

    #[test]
    fn predict() {
        //let x: ColumnFeature<_> = Vec2D::from_slice(&[1, 2, 3, 4], 1).into();
        //let y = vec!(-1.0, -1.0, 1.0, 1.0);

        let tree: Tree<ThresholdSplitter<ColumnFeature<Vec2D<i32>>>, ConstMean<_, Vec<_>>> = Tree {
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
*/