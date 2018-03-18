use std::f64;
use std::marker::PhantomData;
use std::ops;

use rand::{Rand, Rng};
use rand::distributions::{IndependentSample, Range};

use super::FeatureExtractor;

/// Features are equivalent to data columns.
#[derive(Debug)]
pub struct SelectFeature<T: ?Sized> {
    col: usize,
    _p: PhantomData<T>,
}

impl<T> FeatureExtractor for SelectFeature<T>
    where T: Copy
{
    type Xi = [T];
    type Fi = T;

    fn new_random<R: Rng>(x: &Self::Xi, rng: &mut R) -> Self {
        let col = rng.gen_range(0, x.len());
        SelectFeature {
            col,
            _p: PhantomData,
        }
    }

    fn extract(&self, x: &Self::Xi) -> Self::Fi {
        x[self.col]
    }
}

/// Defines a feature as the linear combination of two sample columns.
#[derive(Debug)]
struct Rot2DFeature<T> {
    ia: usize,
    ib: usize,
    wa: f64,
    wb: f64,
    _p: PhantomData<T>,
}

impl<T> FeatureExtractor for Rot2DFeature<T>
    where T: ops::Mul<Output=T> + ops::Add<Output=T> + Copy + Rand + PartialOrd,
          T: Into<f64>
{
    type Xi = [T];
    type Fi = f64;

    fn new_random<R: Rng>(x: &Self::Xi, rng: &mut R) -> Self {
        let mut range = Range::new(0, x.len());
        let angle: f64 = rng.gen();
        debug_assert!(angle >= 0.0);
        debug_assert!(angle <= 1.0);
        Rot2DFeature {
            ia: range.ind_sample(rng),
            ib: range.ind_sample(rng),
            wa: angle.cos(),
            wb: angle.sin(),
            _p: PhantomData,
        }
    }

    fn extract(&self, x: &Self::Xi) -> Self::Fi {
        x[self.ia].into() * self.wa + x[self.ib].into() * self.wb
    }
}


#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use super::*;
    use vec2d::Vec2D;

    #[test]
    fn select() {
        let n_columns = 4;
        let x: Vec2D<i32> = Vec2D::from_slice(&vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), n_columns);

        let sf = SelectFeature {col: 1, _p: PhantomData};
        assert_eq!(sf.extract(&x[1]), 6);

        let sf = SelectFeature::new_random(&x[0], &mut thread_rng());
        assert!(sf.col >= 0);
        assert!(sf.col < n_columns);
    }

    #[test]
    fn rot2d() {
        let n_columns = 4;
        let x: Vec2D<i32> = Vec2D::from_slice(&vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), n_columns);
        let y = vec!(1, 2, 1, 2, 11, 12, 11, 12);

        let sf = Rot2DFeature {ia: 0, ib: 1, wa: 0.8, wb: 0.2, _p: PhantomData};
        assert_eq!(sf.extract(&x[1]), 5.2);

        let sf = Rot2DFeature::new_random(&x[0], &mut thread_rng());
        assert!(sf.ia >= 0);
        assert!(sf.ia < n_columns);
        assert!(sf.ib >= 0);
        assert!(sf.ib < n_columns);
        assert!((sf.wa * sf.wa + sf.wb * sf.wb - 1.0).abs() < 1e-9);
    }
}
