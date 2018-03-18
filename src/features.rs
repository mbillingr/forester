use std::f64;
use std::marker::PhantomData;
use std::ops;

use rand::{Rand, Rng};
use rand::distributions::{IndependentSample, Range};

use vec2d::Vec2D;
use super::FeatureSet;
use super::FixedLength;
use super::Sample;
use super::Shape2D;
use super::OutcomeVariable;

/// Features are equivalent to data columns.
#[derive(Debug)]
pub struct ColumnFeature<Container> {
    data: Container
}

impl<C> From<C> for ColumnFeature<C> {
    fn from(data: C) -> Self {
        ColumnFeature {
            data
        }
    }
}

impl<T> FeatureSet for ColumnFeature<Vec2D<T>>
    where [T]: Sample<Theta=usize, Feature=T>,
          T: PartialOrd + Copy
{
    type Item = [T];

    fn n_samples(&self) -> usize {
        self.data.len()
    }

    fn get_sample(&self, n: usize) -> &Self::Item {
        &self.data[n]
    }

    fn random_feature<R: Rng>(&self, rng: &mut R) -> <Self::Item as Sample>::Theta {
        rng.gen_range(0, self.data.n_columns())
    }

    fn minmax(&self, theta: &<Self::Item as Sample>::Theta) -> Option<(<Self::Item as Sample>::Feature, <Self::Item as Sample>::Feature)> {
        let mut samples = self.data.into_iter();
        let (mut min, mut max) = match samples.next() {
            None => return None,
            Some(x) => {
                let f = x.get_feature(theta);
                (f, f)
            }
        };

        for x in samples {
            let f = x.get_feature(theta);
            if f > max {
                max = f;
            }
            if f < min {
                min = f;
            }
        }

        Some((min, max))
    }

    fn for_each_mut<F: FnMut(&Self::Item)>(&self, mut f: F) {
        for s in self.data.into_iter() {
            f(s)
        }
    }
}

impl<T> Sample for [T]
    where T: Copy
{
    type Theta = usize;
    type Feature = T;
    fn get_feature(&self, theta: &Self::Theta) -> Self::Feature {
        self[*theta]
    }
}

impl<T> OutcomeVariable for [T] {
    type Item = T;

    fn n_samples(&self) -> usize {
        self.len()
    }

    fn for_each_mut<F: FnMut(&Self::Item)>(&self, mut f: F) {
        for y in self.iter() {
            f(y)
        }
    }
}

impl<T> OutcomeVariable for Vec<T> {
    type Item = T;

    fn n_samples(&self) -> usize {
        self.len()
    }

    fn for_each_mut<F: FnMut(&Self::Item)>(&self, mut f: F) {
        for y in self.iter() {
            f(y)
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use super::*;
    use vec2d::Vec2D;

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
    }
}
