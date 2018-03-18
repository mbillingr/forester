
use std::marker::PhantomData;

use super::FixedLength;
use super::SplitCriterion;


struct VarCriterion<K, T: ?Sized> {
    _p: PhantomData<(K, T)>
}

impl<'a, T: 'a, K: 'a> SplitCriterion<'a> for VarCriterion<K, T>
    where &'a T: IntoIterator<Item=&'a K> + FixedLength,
          K: Into<f64> + Copy,
          T: ?Sized
{
    type Y = T;
    type C = f64;

    fn calc_presplit(y: &'a Self::Y) -> f64 {
        let mean: f64 = y.into_iter().map(|yi| (*yi).into()).sum::<f64>() / y.len() as f64;
        y.into_iter().map(|yi| (*yi).into() ).map(|yi| yi * yi).sum::<f64>() / y.len() as f64 - mean * mean
    }

    fn calc_postsplit(yl: &'a Self::Y, yr: &'a Self::Y) -> f64 {
        let a = yl.len() as f64;
        let b = yr.len() as f64;
        (Self::calc_presplit(yl) * a + Self::calc_presplit(yr) * b) / (a + b)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_criterion() {
        let y = vec!(1, 2, 1, 2, 11, 12, 11, 12);

        let c = VarCriterion::calc_presplit(y.as_slice());
        assert_eq!(c, 25.25);

        let c = VarCriterion::calc_postsplit(&y[..4], &y[4..]);
        assert_eq!(c, 0.25);

        let y = vec!(1, 2, 1, 2, 1, 2, 11, 12);

        let c = VarCriterion::calc_postsplit(&y[..6], &y[6..]);
        assert_eq!(c, 0.25);
    }
}
