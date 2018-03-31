use std::cmp;
use std::f64::consts::PI;

use rand::Rng;
//use rand::distributions::{IndependentSample, Range};

use super::Feature;
use super::Real;

use get_item::GetItem;

#[derive(Debug)]
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

    fn random<R: Rng>(x: &X, rng: &mut R) -> Self::Theta {
        rng.gen_range(0, x.n_items())
    }
}

#[derive(Debug)]
pub struct Mix2;

impl<X> Feature<X> for Mix2
where X: GetItem,
      X::Item: Clone + Into<Real>,
{
    type Theta = (usize, usize, Real, Real);
    type F = Real;

    fn get_feature(x: &X, theta: &Self::Theta) -> Real {
        let a: Real = x.get_item(theta.0).clone().into();
        let b: Real = x.get_item(theta.1).clone().into();
        let alpha: Real = theta.2;
        let beta: Real = theta.3;
        a * alpha + b * beta
    }

    fn random<R: Rng>(x: &X, rng: &mut R) -> Self::Theta {
        let a = rng.gen_range(0, x.n_items());
        let mut b = rng.gen_range(0, x.n_items() - 1);
        if b == a {
            b = a + 1;
        }
        let phi = rng.gen::<Real>() * 2.0 * PI;
        //let phi: Real = 0.0;
        let alpha = phi.cos();
        let beta = phi.sin();
        //let alpha = rng.gen::<Real>();
        //let beta = rng.gen::<Real>();
        (a, b, alpha, beta)
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

        assert_eq!(CS::get_feature(&x, &(0, 1, 0.5, 0.5)), 1.5);
        assert_eq!(CS::get_feature(&x, &(1, 3, 0.5, 0.5)), 3.0);
        assert_eq!(CS::get_feature(&x, &(4, 3, 0.1, 0.9)), 4.1);
    }
}
