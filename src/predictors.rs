
use std::f64;
use std::iter;
use std::ops;
use std::marker::PhantomData;

use super::DataSet;
use super::FixedLength;
use super::Float;
use super::LeafPredictor;
use super::ProbabilisticLeafPredictor;
use super::Sample;

use array_ops::Dot;

#[derive(Debug)]
pub struct ConstMean<T: Float, S: ?Sized> {
    value: T,
    _p: PhantomData<S>,
}

impl<T: Float, S: ?Sized> ConstMean<T, S> {
    pub fn new(value: T) -> Self {
        ConstMean {
            value,
            _p: PhantomData
        }
    }
}

impl<S: Sample> LeafPredictor for ConstMean<S::Y, S>
    where S::Y: Float + Copy + iter::Sum + ops::Div<Output=S::Y>
{
    type S = S;
    type D = [S];

    fn predict(&self, s: &<Self::S as Sample>::X) -> <Self::S as Sample>::Y {
        self.value
    }

    fn fit(data: &Self::D) -> Self {
        let sum: <Self::S as Sample>::Y = data.iter().map(|s| s.get_y()).sum();
        let n = <Self::S as Sample>::Y::from_usize(data.len());
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

    fn fit(data: &Self::D) -> Self {
        unimplemented!("LinearRegression::fit()")
    }
}

#[derive(Debug)]
pub struct ConstGaussian<S: ?Sized> {
    mean: f64,
    variance: f64,

    _p: PhantomData<S>,
}

impl<S: Sample<Y=f64>> LeafPredictor for ConstGaussian<S>
{
    type S = S;
    type D = [S];

    fn predict(&self, x: &<Self::S as Sample>::X) -> <Self::S as Sample>::Y {
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

        let n = S::Y::from_usize(data.len());

        let mean = sum / n;
        let variance = ssum / n - mean * mean;

        ConstGaussian {
            mean,
            variance,
            _p: PhantomData,
        }
    }
}

impl<S: Sample<Y=f64>> ProbabilisticLeafPredictor for ConstGaussian<S>
{
    fn prob(&self, s: &Self::S) -> f64 {
        let y = s.get_y();
        f64::exp(-(y - self.mean).powi(2) / (2.0 * self.variance)) / (2.0 * f64::consts::PI * self.variance).sqrt()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use vec2d::Vec2D;
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
