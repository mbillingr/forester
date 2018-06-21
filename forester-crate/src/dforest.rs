//! Deterministic Forest module
//!
//! A deterministic forest is an ensemble of deterministic decision trees.

use std::fmt;

use data::{SampleDescription, TrainingData};
use dtree::{DeterministicTree, DeterministicTreeBuilder};
use iter_mean::IterMean;
use split::SplitFinder;

/// An ensemble of deterministic decision trees.
pub struct DeterministicForest<Sample>
    where Sample: SampleDescription
{
    estimators: Vec<DeterministicTree<Sample>>
}

impl<Sample: SampleDescription> fmt::Debug for DeterministicForest<Sample>
    where Sample::ThetaLeaf: fmt::Debug,
          Sample::ThetaSplit: fmt::Debug,
          Sample::Feature: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Forest: {:#?}", self.estimators)
    }
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
}

/// Fit a `DeterministicForest` to `TrainingData`.
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
    use dtree::Node;
    use testdata::Sample;

    #[test]
    fn fmt() {
        let forest: DeterministicForest<Sample<_, _>> = DeterministicForest {
            estimators: vec![
                DeterministicTree::new_with_nodes(vec![
                    Node::Split { theta: 1, threshold: 2.3, left: 1, right: 2 },
                    Node::Leaf(4.5),
                    Node::Invalid,
                ]),
                DeterministicTree::new_with_nodes(vec![
                    Node::Split { theta: 1, threshold: 2.3, left: 1, right: 2 },
                    Node::Leaf(4.5),
                    Node::Invalid,
                ])
            ]
        };

        let formatted = format!("{:?}", forest);

        let tree_expected = "    Tree:\n    (1) <= 2.3\n     +-- 4.5\n     +-- *** Invalid ***";

        assert_eq!(formatted, format!("Forest: [\n{},\n{}\n]", tree_expected, tree_expected));
    }
}
