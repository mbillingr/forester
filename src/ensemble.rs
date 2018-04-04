use std::marker::PhantomData;

use super::LearnerMut;
use super::Predictor;
use super::Sample;

use array_ops::IterMean;

/// Generic decision forest.
#[derive(Debug)]
pub struct Ensemble<X, Z, P: Predictor<X, Output=Z>>
{
    estimators: Vec<P>,
    _p: PhantomData<(X, Z)>,
}

impl<X, Z, P: Predictor<X, Output=Z>> Predictor<X> for Ensemble<X, Z, P>
    where Z: IterMean<Z>
{
    type Output = Z;
    fn predict(&self, x: &X) -> Z {
        IterMean::mean(self.estimators.iter().map(|tree| tree.predict(x)))
    }
}


#[derive(Debug)]
pub struct EnsembleBuilder<Z, S: Sample, B: LearnerMut<S, P>, P: Predictor<S::X, Output=Z>> {
    n_estimators: usize,
    estimator_builder: B,
    _p: PhantomData<(Z, P, S)>,
}

impl<Z, S: Sample, B: LearnerMut<S, P>, P: Predictor<S::X, Output=Z>> EnsembleBuilder<Z, S, B, P> {
    pub fn new(n_estimators: usize, estimator_builder: B) -> EnsembleBuilder<Z, S, B, P> {
        EnsembleBuilder {
            n_estimators,
            estimator_builder,
            _p: PhantomData
        }
    }
}

impl<Z, S: Sample, B: LearnerMut<S, P>, P: Predictor<S::X, Output=Z>> Default for EnsembleBuilder<Z, S, B, P> {
    fn default() -> EnsembleBuilder<Z, S, B, P> {
        EnsembleBuilder {
            n_estimators: 10,
            estimator_builder: B::default(),
            _p: PhantomData
        }
    }
}

impl<Z, S: Sample, B: LearnerMut<S, P>, P: Predictor<S::X, Output=Z>> LearnerMut<S,  Ensemble<S::X, Z, P>> for EnsembleBuilder<Z, S, B, P>
{
    fn fit(&self, data: &mut [S]) -> Ensemble<S::X, Z, P> {
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensemble() {
        use rand::ThreadRng;
        use d_tree::DeterministicTreeBuilder;
        use splitters::BestRandomSplit;
        use predictors::ConstMean;
        use splitters::ThresholdSplitter;
        use criteria::VarCriterion;
        use datasets::TupleSample;
        use features::ColumnSelect;

        let x = vec![[1], [2], [3],    [7], [8], [9]];
        let y = vec![5.0, 5.0, 5.0,    2.0, 2.0, 2.0];

        let mut data: Vec<_> = x
            .into_iter()
            .zip(y.into_iter())
            .map(|(x, y)| TupleSample::<ColumnSelect, _, _>::new(x, y))
            .collect();

        let estimator_builder: DeterministicTreeBuilder<BestRandomSplit<ThresholdSplitter<_>, VarCriterion<_>, ThreadRng>, ConstMean<_>> = DeterministicTreeBuilder::default();
        let builder = EnsembleBuilder::new(4, estimator_builder);

        let model = builder.fit(&mut data);

        assert_eq!(model.estimators.len(), 4);

        assert_eq!(model.predict(&[1]), 5.0);
        assert_eq!(model.predict(&[8]), 2.0);
    }

}
