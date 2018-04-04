
use std::ops;
use std::marker::PhantomData;

use super::LeafFitter;
use super::LeafPredictor;
use super::ProbabilisticLeafPredictor;
use super::Real;
use super::RealConstants;
use super::Sample;

use array_ops::Dot;

#[derive(Debug)]
pub struct ConstMean<S> {
    value: Real,
    _p: PhantomData<S>,
}

impl<S> ConstMean<S> {
    pub fn new(value: Real) -> Self {
        ConstMean {
            value,
            _p: PhantomData
        }
    }
}

impl<S: Sample<Y=Real>> LeafPredictor<S> for ConstMean<S>
{
    type Output = S::Y;

    fn predict(&self, _x: &S::X) -> Self::Output {
        self.value
    }
}

impl<S: Sample<Y=Real>> LeafFitter<S> for ConstMean<S> {
    type Output = ConstMean<S>;

    fn fit(data: &[S]) -> Self {
        let sum: S::Y = data.iter().map(|s| s.get_y()).sum();
        let n = data.len() as Real;
        ConstMean {
            value: sum / n,
            _p: PhantomData,
        }
    }
}

/// Linear regression with intercept
#[derive(Debug)]
pub struct LinearRegression<T, S> {
    intercept: T,
    weights: Vec<T>,
    _p: PhantomData<S>,
}

impl<S> LeafPredictor<S> for LinearRegression<S::Y, S>
    where S: Sample,
          S::Y: Copy + ops::Mul<Output=S::Y> + ops::Add<Output=S::Y>,
          S::X: Dot<[S::Y], Output=S::Y>
{
    type Output = S::Y;

    /// predicted value
    fn predict(&self, x: &S::X) -> Self::Output {
        self.intercept + x.dot(&self.weights)
    }
}

impl<S> LeafFitter<S> for LinearRegression<S::Y, S>
    where S: Sample,
          S::Y: Copy + ops::Mul<Output=S::Y> + ops::Add<Output=S::Y>,
          S::X: Dot<[S::Y], Output=S::Y>
{
    type Output = LinearRegression<S::Y, S>;

    fn fit(_data: &[S]) -> Self {
        unimplemented!("LinearRegression::fit()")
    }
}

#[derive(Debug)]
pub struct ConstGaussian<S> {
    mean: Real,
    variance: Real,

    _p: PhantomData<S>,
}

impl<S: Sample<Y=Real>> LeafPredictor<S> for ConstGaussian<S>
{

    type Output = S::Y;

    fn predict(&self, _x: &S::X) -> Self::Output {
        self.mean
    }
}

impl<S: Sample<Y=Real>> LeafFitter<S> for ConstGaussian<S> {
    type Output = ConstGaussian<S>;

    fn fit(data: &[S]) -> Self {
        let mut sum = S::Y::zero();
        let mut ssum = S::Y::zero();

        for sample in data {
            let yi = sample.get_y();
            sum += yi;
            ssum += yi * yi;
        }

        let n = data.len() as Real;

        let mean = sum / n;
        let variance = ssum / n - mean * mean;

        ConstGaussian {
            mean,
            variance,
            _p: PhantomData,
        }
    }
}

impl<S: Sample<Y=Real>> ProbabilisticLeafPredictor<S> for ConstGaussian<S>
{
    fn prob(&self, s: &S) -> Real {
        let y = s.get_y();
        Real::exp(-(y - self.mean).powi(2) / (2.0 * self.variance)) / (2.0 * Real::pi() * self.variance).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct CategoricalProbabilities {
    counts: Vec<usize>,
    total: usize,
}

impl CategoricalProbabilities {
    pub fn new() -> Self {
        CategoricalProbabilities {
            counts: Vec::new(),
            total: 0,
        }
    }

    pub fn add_one(&mut self, c: u8) {
        let c = c as usize;
        if c >= self.counts.len() {
            self.counts.resize(c + 1, 0);  // TODO: make number of a-priori counts a regularization parameter
        }
        self.counts[c] += 1;
        self.total += 1;
    }

    pub fn add(&mut self, other: &Self) {
        if other.counts.len() > self.counts.len() {
            self.counts.resize(other.counts.len(), 0);
        }

        for (a, b) in self.counts.iter_mut().zip(other.counts.iter()) {
            *a += *b;
            self.total += *b;
        }
    }

    pub fn probs(&self) -> Vec<Real> {
        self.counts.iter().map(|&c| c as Real / self.total as Real).collect()
    }

    pub fn apply<F: FnMut(Real)>(&self, mut f: F) {
        for &c in &self.counts {
            f(c as Real / self.total as Real);
        }
    }

    pub fn prob(&self, c: u8) -> Real {
        if (c as usize) < self.counts.len() {
            (self.counts[c as usize] as Real) / (self.total as Real)
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct ClassPredictor<S> {
    counts: CategoricalProbabilities,
    _p: PhantomData<S>,
}

impl<S: Sample<Y=u8>> LeafPredictor<S> for ClassPredictor<S>
{
    type Output = CategoricalProbabilities;

    fn predict(&self, _x: &S::X) -> Self::Output {
        self.counts.clone()
    }
}

impl<S: Sample<Y=u8>> LeafFitter<S> for ClassPredictor<S> {
    type Output = ClassPredictor<S>;

    fn fit(data: &[S]) -> Self {
        let mut counts = CategoricalProbabilities::new();

        for sample in data {
            let yi = *sample.get_y();
            counts.add_one(yi);
        }

        ClassPredictor {
            counts,
            _p: PhantomData,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use features::ColumnSelect;
    use datasets::TupleSample;

    #[test]
    fn const_mean() {
        let data: Vec<TupleSample<ColumnSelect, [();0], _>>;
        data = vec![TupleSample::new([], 1.0),
                    TupleSample::new([], 2.0),
                    TupleSample::new([], 3.0),
                    TupleSample::new([], 4.0)];

        let cg = ConstMean::fit(&data);
        assert_eq!(cg.predict(& []), 2.5);
    }

    #[test]
    fn linear_regression() {
        let data: Vec<TupleSample<ColumnSelect, _, _>>;
        data = vec![TupleSample::new([-1.0, 1.0], 1.0),
                    TupleSample::new([-1.0, 2.0], 2.0),
                    TupleSample::new([1.0, 3.0], 13.0),
                    TupleSample::new([1.0, 4.0], 14.0)];

        let reg: LinearRegression<_, TupleSample<ColumnSelect, _, _>>;
        reg = LinearRegression {
            intercept: 5.0,
            weights: vec!(5.0, 1.0),
            _p: PhantomData,
        };

        assert_eq!(reg.predict(data[0].get_x()), 1.0);
        assert_eq!(reg.predict(data[1].get_x()), 2.0);
        assert_eq!(reg.predict(data[2].get_x()), 13.0);
        assert_eq!(reg.predict(data[3].get_x()), 14.0);
    }

    #[test]
    fn const_gaussian() {
        let data: Vec<TupleSample<ColumnSelect, _, _>>;
        data = vec![TupleSample::new([1], 1.0),
                    TupleSample::new([2], 2.0),
                    TupleSample::new([3], 1.0),
                    TupleSample::new([4], 2.0)];

        let cg = ConstGaussian::fit(&data);
        assert_eq!(cg.predict(&[99]), 1.5);
        assert_eq!(cg.prob(&TupleSample::new([0], 0.0)), 0.008863696823876015);
        assert_eq!(cg.prob(&TupleSample::new([0], 0.5)), 0.107981933026376130);
        assert_eq!(cg.prob(&TupleSample::new([0], 1.0)), 0.483941449038286730);
        assert_eq!(cg.prob(&TupleSample::new([0], 1.5)), 0.797884560802865400);
        assert_eq!(cg.prob(&TupleSample::new([0], 2.0)), 0.483941449038286730);
        assert_eq!(cg.prob(&TupleSample::new([0], 2.5)), 0.107981933026376130);
        assert_eq!(cg.prob(&TupleSample::new([0], 3.0)), 0.008863696823876015);
    }
}
