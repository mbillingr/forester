use std::cmp;
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
      X::Item: Clone + cmp::PartialOrd,
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

    #[test]
    fn column_select() {
        let n_columns = 4;
        let x = vec!(1, 2, 33, 4, 5);

        type CS = ColumnSelect;

        assert_eq!(CS::get_feature(&x, &0), 1);
        assert_eq!(CS::get_feature(&x, &2), 33);
        assert_eq!(CS::get_feature(&x, &4), 5);
    }

    #[test]
    fn mix2() {
        let n_columns = 4;
        let x = vec!(1, 2, 33, 4, 5);

        type CS = Mix2;

        assert_eq!(CS::get_feature(&x, &(0, 1, 0.5)), 1.5);
        assert_eq!(CS::get_feature(&x, &(1, 3, 0.5)), 3.0);
        assert_eq!(CS::get_feature(&x, &(4, 3, 0.1)), 4.1);
    }
}
