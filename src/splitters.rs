use std::cmp;
use std::marker::PhantomData;
use std::ops;

use rand::Rng;
//use rand::distributions::{IndependentSample, Range};
use rand::distributions::range::SampleRange;

use super::DataSet;
use super::DeterministicSplitter;
use super::Sample;
use super::Side;
use super::Splitter;

/// Use a simple threshold for deterministic split.
pub struct ThresholdSplitter<D>
    where D: DataSet
{
    theta: <D::Item as Sample>::Theta,
    threshold: <D::Item as Sample>::F,
}

impl<D> ThresholdSplitter<D>
    where D: DataSet
{
    pub fn new(theta: <D::Item as Sample>::Theta, threshold: <D::Item as Sample>::F) -> Self {
        ThresholdSplitter {
            theta,
            threshold
        }
    }
}

impl<D> Splitter for ThresholdSplitter<D>
    where D: DataSet,
          //<F::Item as Sample>::Feature: SampleRange + PartialOrd
{
    type D = D;

    fn new_random<R: Rng>(x: &D, rng: &mut R) -> Self
    {
        unimplemented!()
        /*let theta = x.random_feature(rng);
        let (low, high) = x.minmax(&theta).expect("Could not find min/max for feature");
        let threshold = rng.gen_range(low, high);
        ThresholdSplitter {
            theta,
            threshold,
        }*/
    }
}

impl<D> DeterministicSplitter for ThresholdSplitter<D>
    where D: DataSet,
          //<D::Item as Sample>::F: SampleRange + PartialOrd
{
    fn split(&self, s: &<Self::D as DataSet>::Item) -> Side {
        let f = s.get_feature(&self.theta);
        if f <= self.threshold {
            Side::Left
        } else {
            Side::Right
        }
    }
}