
use std::ops;
use std::marker::PhantomData;

use super::LeafPredictor;
use super::ProbabilisticLeafPredictor;
use super::Real;
use super::RealConstants;
use super::Sample;

use array_ops::Dot;

#[derive(Debug)]
pub struct ConstMean<S: ?Sized> {
    value: Real,
    _p: PhantomData<S>,
}

impl<S: ?Sized> ConstMean<S> {
    pub fn new(value: Real) -> Self {
        ConstMean {
            value,
            _p: PhantomData
        }
    }
}

impl<S: Sample<Y=Real>> LeafPredictor for ConstMean<S>
    //where S::Y: Real + Copy + iter::Sum + ops::Div<Output=S::Y>
{
    type S = S;
    type D = [S];

    fn predict(&self, _s: &<Self::S as Sample>::X) -> <Self::S as Sample>::Y {
        self.value
    }

    fn fit(data: &Self::D) -> Self {
        let sum: <Self::S as Sample>::Y = data.iter().map(|s| s.get_y()).sum();
        let n = data.len() as Real;
        ConstMean {
            value: sum / n,
            _p: PhantomData,
        }
    }
}

/// Linear regression with intercept
#[derive(Debug)]
pub struct LinearRegression<T, S: ?Sized> {
    intercept: T,
    weights: Vec<T>,
    _p: PhantomData<S>,
}

impl<S: Sample> LeafPredictor for LinearRegression<S::Y, S>
    where S::Y: Copy + ops::Mul<Output=S::Y> + ops::Add<Output=S::Y>,
          S::X: Dot<[S::Y], Output=S::Y>
{
    type S = S;
    type D = [S];

    /// predicted value
    fn predict(&self, x: &<Self::S as Sample>::X) -> <Self::S as Sample>::Y {
        self.intercept + x.dot(&self.weights)
    }

    fn fit(_data: &Self::D) -> Self {
        unimplemented!("LinearRegression::fit()")
    }
}

#[derive(Debug)]
pub struct ConstGaussian<S: ?Sized> {
    mean: Real,
    variance: Real,

    _p: PhantomData<S>,
}

impl<S: Sample<Y=Real>> LeafPredictor for ConstGaussian<S>
{
    type S = S;
    type D = [S];

    fn predict(&self, _x: &<Self::S as Sample>::X) -> <Self::S as Sample>::Y {
        self.mean
    }

    fn fit(data: &Self::D) -> Self {
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

impl<S: Sample<Y=Real>> ProbabilisticLeafPredictor for ConstGaussian<S>
{
    fn prob(&self, s: &Self::S) -> Real {
        let y = s.get_y();
        Real::exp(-(y - self.mean).powi(2) / (2.0 * self.variance)) / (2.0 * Real::pi() * self.variance).sqrt()
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

        assert_eq!(reg.predict(&data[0].get_x()), 1.0);
        assert_eq!(reg.predict(&data[1].get_x()), 2.0);
        assert_eq!(reg.predict(&data[2].get_x()), 13.0);
        assert_eq!(reg.predict(&data[3].get_x()), 14.0);
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
