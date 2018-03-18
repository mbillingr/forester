
use super::LeafPredictor;
use super::Splitter;
/*
/// A decision tree node. Can be either a split node with a Splitter and two children, or a leaf
/// node with a LeafPredictor.
#[derive(Debug)]
enum Node<S: Splitter, L: LeafPredictor>
{
    Invalid,  // placeholder used during tree construction
    Split{split: S, left: usize, right: usize},
    Leaf(L),
    Hybrid{predictor: L, split: S, left: usize, right: usize},
}
*/