use std::cmp;
use std::marker::PhantomData;
use std::ops;

use rand::{Rand, Rng};
use rand::distributions::{IndependentSample, Range};
use rand::distributions::range::SampleRange;

use super::DeterministicSplitter;
use super::FixedLength;
use super::FeatureSet;
use super::Sample;
use super::Side;
use super::Splitter;

/// Use a simple threshold for deterministic split.
pub struct ThresholdSplitter<F>
    where F: FeatureSet
{
    theta: <F::Item as Sample>::Theta,
    threshold: <F::Item as Sample>::Feature,
}


impl<F> Splitter for ThresholdSplitter<F>
    where F: FeatureSet,
          <F::Item as Sample>::Feature: SampleRange + PartialOrd
{
    type F = F;

    fn new_random<R: Rng>(x: &F, rng: &mut R) -> Self
    {
        let theta = x.random_feature(rng);
        let (low, high) = x.minmax(&theta).expect("Could not find min/max for feature");
        let threshold = rng.gen_range(low, high);
        ThresholdSplitter {
            theta,
            threshold,
        }
    }
}

impl<F> DeterministicSplitter for ThresholdSplitter<F>
    where F: FeatureSet,
          <F::Item as Sample>::Feature: SampleRange + PartialOrd
{
    fn split(&self, x: &F::Item) -> Side {
        let f = x.get_feature(&self.theta);
        if f <= self.threshold {
            Side::Left
        } else {
            Side::Right
        }
    }
}
