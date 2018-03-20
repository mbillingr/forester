use std::iter;
use std::marker::PhantomData;
use std::ops;

use super::DataSet;
use super::DeterministicSplitter;
use super::LearnerMut;
use super::LeafPredictor;
use super::Predictor;
use super::Sample;
use super::Splitter;
use super::SplitFitter;

use d_tree::{DeterministicTree, DeterministicTreeBuilder};

/// Generic decision forest.
#[derive(Debug)]
pub struct Ensemble<X, Y, P: Predictor<X, Y>>
{
    estimators: Vec<P>,
    _p: PhantomData<(X, Y)>,
}

impl<X, Y, P: Predictor<X, Y>> Predictor<X, Y> for Ensemble<X, Y, P>
    where Y: iter::Sum + From<usize> + ops::Div<Output=Y>
{
    fn predict(&self, x: &X) -> Y {
        self.estimators.iter().map(|tree| tree.predict(x)).sum::<Y>() / Y::from(self.estimators.len())
    }
}


#[derive(Debug)]
struct EnsembleBuilder<D: ?Sized + DataSet, B: LearnerMut<D, P>, P: Predictor<D::X, D::Y>> {
    n_estimators: usize,
    estimator_builder: B,
    _p: PhantomData<(P, D)>,
}

impl<D: ?Sized + DataSet, B: LearnerMut<D, P>, P: Predictor<D::X, D::Y>> LearnerMut<D,  Ensemble<D::X, D::Y, P>> for EnsembleBuilder<D, B, P>
{
    fn fit(&self, data: &mut D) -> Ensemble<D::X, D::Y, P> {
        let mut estimators = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            estimators.push(self.estimator_builder.fit(data))
        }

        Ensemble {
            estimators,
            _p: PhantomData,
        }
    }
}
