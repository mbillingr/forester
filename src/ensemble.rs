use std::marker::PhantomData;

use super::DataSet;
use super::LearnerMut;
use super::Predictor;

use array_ops::IterMean;

/// Generic decision forest.
#[derive(Debug)]
pub struct Ensemble<X, Z, P: Predictor<X, Z>>
{
    estimators: Vec<P>,
    _p: PhantomData<(X, Z)>,
}

impl<X, Z, P: Predictor<X, Z>> Predictor<X, Z> for Ensemble<X, Z, P>
    where Z: IterMean<Z>
{
    fn predict(&self, x: &X) -> Z {
        IterMean::mean(self.estimators.iter().map(|tree| tree.predict(x)))
    }
}


#[derive(Debug)]
pub struct EnsembleBuilder<Z, D: ?Sized + DataSet, B: LearnerMut<D, P>, P: Predictor<D::X, Z>> {
    n_estimators: usize,
    estimator_builder: B,
    _p: PhantomData<(Z, P, D)>,
}

impl<Z, D: ?Sized + DataSet, B: LearnerMut<D, P>, P: Predictor<D::X, Z>> EnsembleBuilder<Z, D, B, P> {
    pub fn new(n_estimators: usize, estimator_builder: B) -> EnsembleBuilder<Z, D, B, P> {
        EnsembleBuilder {
            n_estimators,
            estimator_builder,
            _p: PhantomData
        }
    }
}

impl<Z, D: ?Sized + DataSet, B: LearnerMut<D, P>, P: Predictor<D::X, Z>> Default for EnsembleBuilder<Z, D, B, P> {
    fn default() -> EnsembleBuilder<Z, D, B, P> {
        EnsembleBuilder {
            n_estimators: 10,
            estimator_builder: B::default(),
            _p: PhantomData
        }
    }
}

impl<Z, D: ?Sized + DataSet, B: LearnerMut<D, P>, P: Predictor<D::X, Z>> LearnerMut<D,  Ensemble<D::X, Z, P>> for EnsembleBuilder<Z, D, B, P>
{
    fn fit(&self, data: &mut D) -> Ensemble<D::X, Z, P> {
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
