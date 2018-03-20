use std::marker::PhantomData;

//use rand::Rng;
//use rand::distributions::{IndependentSample, Range};
//use rand::distributions::range::SampleRange;

use super::DataSet;
use super::DeterministicSplitter;
use super::Feature;
use super::Sample;
use super::Side;
use super::SplitCriterion;
use super::SplitFitter;
use super::Splitter;

/// Use a simple threshold for deterministic split.
pub struct ThresholdSplitter<D>
    where D: ?Sized + DataSet
{
    theta: <D::Item as Sample>::Theta,
    threshold: <D::Item as Sample>::F,
}

impl<D> ThresholdSplitter<D>
    where D: ?Sized + DataSet
{
    pub fn new(theta: <D::Item as Sample>::Theta, threshold: <D::Item as Sample>::F) -> Self {
        ThresholdSplitter {
            theta,
            threshold
        }
    }
}

impl<D> Splitter for ThresholdSplitter<D>
    where D: ?Sized + DataSet,
          //<F::Item as Sample>::Feature: SampleRange + PartialOrd
{
    type D = D;

    fn new_random(_x: &D) -> Self
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

    fn theta(&self) -> &<Self::D as DataSet>::Theta {
        &self.theta
    }
}

impl<D> DeterministicSplitter for ThresholdSplitter<D>
    where D: ?Sized + DataSet,
          //<D::Item as Sample>::F: SampleRange + PartialOrd
{
    fn split(&self, x: &<Self::D as DataSet>::X) -> Side {
        let f = D::FX::get_feature(x, &self.theta);
        if f <= self.threshold {
            Side::Left
        } else {
            Side::Right
        }
    }
}

struct BestRandomSplit<S, C> {
    n_splits: usize,
    _p: PhantomData<(S, C)>,
}

impl<S: DeterministicSplitter, C: SplitCriterion<D=S::D>> SplitFitter for BestRandomSplit<S, C> {
    type D = S::D;
    type Split = S;
    type Criterion = C;
    fn find_split(&self, data: &Self::D) -> Option<(Self::Split, Vec<usize>, Vec<usize>)> {
        /*let mut best_criterion = None;
        let mut best_split = None;

        let parent_criterion = C::calc_presplit(data);

        for _ in 0..self.n_splits {
            let split: S = Self::Split::new_random(data);

            data.sort_by_feature(&split.theta());

            // TODO: find split index, compare criterions

            // TODO: what should we finally return?
        }*/
        unimplemented!()
    }
}
