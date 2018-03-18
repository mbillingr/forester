
use std::f64;
use std::ops;
use std::marker::PhantomData;

use super::FixedLength;
use super::LeafPredictor;
use super::ProbabilisticLeafPredictor;

#[derive(Debug)]
pub struct ConstMean<A: ?Sized, B: ?Sized> {
    value: f64,

    _a: PhantomData<A>,
    _b: PhantomData<B>,
}

impl<'a, A: 'a, B: 'a> LeafPredictor<'a, f64> for ConstMean<A, B>
    where &'a A: IntoIterator,
          &'a B: IntoIterator<Item=&'a f64>,
          &'a B: FixedLength,
          A: ?Sized,
          B: ?Sized,
{
    type X = A;
    type Y = B;

    fn predict(&self, x: <&'a Self::X as IntoIterator>::Item) -> f64 {
        self.value
    }

    fn fit(x: &'a Self::X, y: &'a Self::Y) -> Self {
        let value = y.into_iter().sum::<f64>() / y.len() as f64;
        ConstMean {
            value,
            _a: PhantomData,
            _b: PhantomData,
        }
    }
}

/// Linear regression with intercept
#[derive(Debug)]
pub struct LinearRegression<'a, X: 'a, Y: 'a, T> {
    intercept: T,
    weights: Vec<T>,
    _p: PhantomData<&'a (X, Y)>,
}

impl<'a, A: 'a, B: 'a, T: 'a> LeafPredictor<'a, T> for LinearRegression<'a, A, B, T>
    where &'a A: IntoIterator<Item=&'a [T]>,
          &'a B: IntoIterator<Item=&'a T>,
          T: ops::Mul<Output=T> + ops::AddAssign + Copy
{
    type X = A;
    type Y = B;

    /// predicted value
    fn predict(&self, x: <&'a Self::X as IntoIterator>::Item) -> T {
        let mut result = self.intercept;
        for (xi, wi) in x.iter().zip(&self.weights) {
            result += *xi * *wi;
        }
        result
    }

    fn fit(x: &Self::X, y: &Self::Y) -> Self {
        unimplemented!("LinearRegression::fit()")
    }
}

#[derive(Debug)]
pub struct ConstGaussian<A: ?Sized, B: ?Sized> {
    mean: f64,
    variance: f64,

    _a: PhantomData<A>,
    _b: PhantomData<B>,
}

impl<'a, A: 'a, B: 'a> LeafPredictor<'a, f64> for ConstGaussian<A, B>
    where &'a A: IntoIterator,
          &'a B: IntoIterator<Item=&'a f64>,
          &'a B: FixedLength,
          A: ?Sized,
          B: ?Sized,
{
    type X = A;
    type Y = B;

    fn predict(&self, x: <&'a Self::X as IntoIterator>::Item) -> f64 {
        self.mean
    }

    fn fit(x: &'a Self::X, y: &'a Self::Y) -> Self {
        let mean = y.into_iter().sum::<f64>() / y.len() as f64;
        let variance = y.into_iter()
            .map(|yi| (yi - mean))
            .map(|yi| yi * yi).sum::<f64>() / y.len() as f64;
        ConstGaussian {
            mean,
            variance,
            _a: PhantomData,
            _b: PhantomData,
        }
    }
}

impl<'a, A: 'a, B: 'a> ProbabilisticLeafPredictor<'a, f64> for ConstGaussian<A, B>
    where &'a A: IntoIterator,
          &'a B: IntoIterator<Item=&'a f64>,
          &'a B: FixedLength,
          A: ?Sized,
          B: ?Sized,
{
    fn prob(&self, x: <&'a Self::X as IntoIterator>::Item, y: <&'a Self::Y as IntoIterator>::Item) -> f64 {
        f64::exp(-(y - self.mean).powi(2) / (2.0 * self.variance)) / (2.0 * f64::consts::PI * self.variance).sqrt()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use vec2d::Vec2D;

    #[test]
    fn const_mean() {
        let x: Vec<()> = Vec::new();
        let y = vec!(1.0, 2.0, 3.0, 4.0);

        let cg = ConstMean::fit(&x, &y);
        assert_eq!(cg.predict(&()), 2.5);
    }

    #[test]
    fn const_gaussian() {
        let x: Vec<()> = Vec::new();
        let y = vec!(1.0, 2.0, 1.0, 2.0);

        let cg = ConstGaussian::fit(&x, &y);
        assert_eq!(cg.predict(&()), 1.5);
        assert_eq!(cg.prob(&(), &0.0), 0.008863696823876015);
        assert_eq!(cg.prob(&(), &0.5), 0.107981933026376130);
        assert_eq!(cg.prob(&(), &1.0), 0.483941449038286730);
        assert_eq!(cg.prob(&(), &1.5), 0.797884560802865400);
        assert_eq!(cg.prob(&(), &2.0), 0.483941449038286730);
        assert_eq!(cg.prob(&(), &2.5), 0.107981933026376130);
        assert_eq!(cg.prob(&(), &3.0), 0.008863696823876015);
    }
}
