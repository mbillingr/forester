
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

#[derive(Debug)]
pub struct ConstMean<T, S: ?Sized> {
    value: T,
    _p: PhantomData<S>,
}

impl<T, S: ?Sized> ConstMean<T, S> {
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
    type D = [S];
    type S = S;

    fn predict(&self, s: <Self::S as Sample>::X) -> <Self::S as Sample>::Y {
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
/*
/// Linear regression with intercept
#[derive(Debug)]
pub struct LinearRegression<X, Y, T> {
    intercept: T,
    weights: Vec<T>,
    _p: PhantomData<(X, Y)>,
}

impl<X, Y, T> LeafPredictor for LinearRegression<X, Y, T>
    where X: FeatureSet<Item=[T]>,
          Y: OutcomeVariable<Item=T>,
          T: ops::Mul<Output=T> + ops::AddAssign + Copy
{
    /*type X = X;
    type Y = Y;*/

    /// predicted value
    fn predict(&self, x: &X::Item) -> T {
        let mut result = self.intercept;
        for (xi, wi) in x.iter().zip(&self.weights) {
            result += *xi * *wi;
        }
        result
    }

    fn fit(_x: &X, _y: &Y) -> Self {
        unimplemented!("LinearRegression::fit()")
    }
}

#[derive(Debug)]
pub struct ConstGaussian<X, Y> {
    mean: f64,
    variance: f64,

    _p: PhantomData<(X, Y)>,
}


impl<X, Y> LeafPredictor for ConstGaussian<X, Y>
    where X: FeatureSet,
          Y: OutcomeVariable<Item=f64>
{
    /*type X = X;
    type Y = Y;*/

    fn predict(&self, _x: &X::Item) -> f64 {
        self.mean
    }

    fn fit(_x: &X, y: &Y) -> Self {
        let mut sum = 0.0;
        let mut ssum = 0.0;
        y.for_each_mut(|yi| {
            sum += yi;
            ssum += yi * yi;
        });
        let mean = sum / y.n_samples() as f64;
        ConstGaussian {
            mean,
            variance: ssum / y.n_samples() as f64 - mean * mean,
            _p: PhantomData,
        }
    }
}

impl<X, Y> ProbabilisticLeafPredictor for ConstGaussian<X, Y>
    where X: FeatureSet,
          Y: OutcomeVariable<Item=f64>
{
    fn prob(&self, _x: &X::Item, y: &Y::Item) -> f64 {
        f64::exp(-(y - self.mean).powi(2) / (2.0 * self.variance)) / (2.0 * f64::consts::PI * self.variance).sqrt()
    }
}
*/

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
        assert_eq!(cg.predict([]), 2.5);
    }
/*
    #[test]
    fn linear_regression() {
        let x: ColumnFeature<_> = Vec2D::from_slice(&[4.0, 3.0, 2.0, 1.0], 1).into();
        let y = vec!(1.0, 2.0, 3.0, 4.0);

        //let cg = LinearRegression::fit(&x, &y);
        let cg: LinearRegression<ColumnFeature<Vec2D<_>>, Vec<_>, _> = LinearRegression {
            intercept: 5.0,
            weights: vec!(-1.0),
            _p: PhantomData
        };

        assert_eq!(cg.predict(x.get_sample(0)), y[0]);
        assert_eq!(cg.predict(x.get_sample(1)), y[1]);
        assert_eq!(cg.predict(x.get_sample(2)), y[2]);
        assert_eq!(cg.predict(x.get_sample(3)), y[3]);
    }

    #[test]
    fn const_gaussian() {
        let x: ColumnFeature<_> = Vec2D::from_slice(&[0], 1).into();
        let y = vec!(1.0, 2.0, 1.0, 2.0);

        let cg = ConstGaussian::fit(&x, &y);
        assert_eq!(cg.predict(&[]), 1.5);
        assert_eq!(cg.prob(&[], &0.0), 0.008863696823876015);
        assert_eq!(cg.prob(&[], &0.5), 0.107981933026376130);
        assert_eq!(cg.prob(&[], &1.0), 0.483941449038286730);
        assert_eq!(cg.prob(&[], &1.5), 0.797884560802865400);
        assert_eq!(cg.prob(&[], &2.0), 0.483941449038286730);
        assert_eq!(cg.prob(&[], &2.5), 0.107981933026376130);
        assert_eq!(cg.prob(&[], &3.0), 0.008863696823876015);
    }*/
}