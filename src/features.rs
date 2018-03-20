use std::cmp;

//use rand::{Rand, Rng};
//use rand::distributions::{IndependentSample, Range};

use super::Feature;
use super::Real;

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
      X::Item: Copy + Into<Real>,
{
    type Theta = (usize, usize, Real);
    type F = Real;

    fn get_feature(x: &X, theta: &Self::Theta) -> Real {
        let alpha: Real = theta.2;
        let a: Real = (*x.get_item(theta.0)).into();
        let b: Real = (*x.get_item(theta.1)).into();
        a * alpha + b * (1.0 - alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_select() {
        let x = vec!(1, 2, 33, 4, 5);

        type CS = ColumnSelect;

        assert_eq!(CS::get_feature(&x, &0), 1);
        assert_eq!(CS::get_feature(&x, &2), 33);
        assert_eq!(CS::get_feature(&x, &4), 5);
    }

    #[test]
    fn mix2() {
        let x = vec!(1, 2, 33, 4, 5);

        type CS = Mix2;

        assert_eq!(CS::get_feature(&x, &(0, 1, 0.5)), 1.5);
        assert_eq!(CS::get_feature(&x, &(1, 3, 0.5)), 3.0);
        assert_eq!(CS::get_feature(&x, &(4, 3, 0.1)), 4.1);
    }
}
