use std::f64;
use std::marker::PhantomData;
use std::ops;

use rand::{Rand, Rng};
use rand::distributions::{IndependentSample, Range};

use super::Feature;
use super::FixedLength;
use super::Sample;
use super::Shape2D;

use get_item::GetItem;

pub struct ColumnSelect;

impl<X> Feature<X> for ColumnSelect
where X: GetItem,
      X::Item: Clone,
{
    type Theta = usize;
    type F = X::Item;

    fn get_feature(x: &X, theta: &usize) -> Self::F {
        x.get_item(*theta).clone()
    }
}

pub struct Mix2;

impl<X> Feature<X> for Mix2
where X: GetItem,
      X::Item: Copy + Into<f64>,
{
    type Theta = (usize, usize, f64);
    type F = f64;

    fn get_feature(x: &X, theta: &Self::Theta) -> f64 {
        let alpha: f64 = theta.2;
        let a: f64 = (*x.get_item(theta.0)).into();
        let b: f64 = (*x.get_item(theta.1)).into();
        a * alpha + b * (1.0 - alpha)
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use super::*;
    use vec2d::Vec2D;
/*
    #[test]
    fn column_feature() {
        let n_columns = 4;
        let x: Vec2D<i32> = Vec2D::from_slice(&vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), n_columns);

        let fs = ColumnFeature::from(x);

        assert_eq!(fs.n_samples(), 3);

        assert_eq!(fs.minmax(&0), Some((1, 9)));
        assert_eq!(fs.minmax(&1), Some((2, 10)));
        assert_eq!(fs.minmax(&2), Some((3, 11)));
        assert_eq!(fs.minmax(&3), Some((4, 12)));

        for _ in 0..100 {
            assert!(fs.random_feature(&mut thread_rng()) < 4);
        }
    }*/
}
