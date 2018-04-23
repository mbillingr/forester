/// Deterministic Forest module
///
/// A deterministic forest is an ensemble of deterministic decision trees.

use data::{SampleDescription, TrainingData};
use dtree::{DeterministicTree, DeterministicTreeBuilder};
use iter_mean::IterMean;
use split::SplitFinder;

pub struct DeterministicForest<Sample>
    where Sample: SampleDescription
{
    estimators: Vec<DeterministicTree<Sample>>
}

/// An ensemble of deterministic decision trees.
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