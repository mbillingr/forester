use std::cmp;
use std::f64;

use rand::Rng;

use array_ops::Partition;

use super::Real;

pub trait RealConstants {
    #[inline(always)] fn pi() -> Real;
    #[inline(always)] fn zero() -> Real {0.0}
    #[inline(always)] fn one() -> Real {1.0}
}

impl RealConstants for Real {
    #[inline(always)] fn pi() -> Real {f64::consts::PI}
}

/// The side of a split
#[derive(Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

pub trait Feature<X> {
    type Theta;
    type F: cmp::PartialOrd + Clone;
    fn get_feature(x: &X, theta: &Self::Theta) -> Self::F;
    fn random<R: Rng>(x: &X, rng: &mut R) -> Self::Theta;
}

pub trait Sample {
    type Theta;
    type F: cmp::PartialOrd + Clone;
    type FX: Feature<Self::X, Theta=Self::Theta, F=Self::F>;
    type X;
    type Y;
    fn get_x(&self) -> &Self::X;
    fn get_y(&self) -> &Self::Y;
}

pub trait DataSet {
    type Theta;
    type F: cmp::PartialOrd + Clone;
    type FX: Feature<Self::X, Theta=Self::Theta, F=Self::F>;
    type X;
    type Y;
    type Item: Sample<Theta=Self::Theta, F=Self::F, FX=Self::FX, X=Self::X, Y=Self::Y>;

    fn n_samples(&self) -> usize;
    fn get(&self, i: usize) -> &Self::Item;

    fn partition_by_split<S: DeterministicSplitter<S=Self::Item>>(&mut self, s: &S) -> usize;

    fn subsets(&mut self, i: usize) -> (&mut Self, &mut Self);

    fn random_feature<R: Rng>(&self, rng: &mut R) -> Self::Theta;

    fn reduce_feature<B, F: FnMut(B, Self::F) -> B>(&self, theta: &Self::Theta, init: B, f: F) -> B;
}

impl<S> DataSet for [S]
    where S: Sample,
//S::F: fmt::Debug,
{
    type Theta = S::Theta;
    type F = S::F;
    type FX = S::FX;
    type X = S::X;
    type Y = S::Y;
    type Item = S;

    fn n_samples(&self) -> usize {
        self.len()
    }

    fn get(&self, i: usize) -> &Self::Item {
        &self[i]
    }

    fn partition_by_split<SP: DeterministicSplitter<S=S>>(&mut self, split: &SP) -> usize {
        self.partition(|sample| split.split(&sample.get_x()) == Side::Left)
    }

    fn subsets(&mut self, i: usize) -> (&mut Self, &mut Self) {
        self.split_at_mut(i)
    }

    fn random_feature<R: Rng>(&self, rng: &mut R) -> Self::Theta {
        S::FX::random(&self[0].get_x(), rng)
    }

    fn reduce_feature<B, F: FnMut(B, Self::F) -> B>(&self, theta: &Self::Theta, init: B, f: F) -> B {
        self.iter().map(|s| Self::FX::get_feature(&s.get_x(), theta)).fold(init, f)
    }
}

/// For comparing splits
pub trait SplitCriterion {
    type S: Sample;
    type C: cmp::PartialOrd + Copy;
    fn calc_presplit(y: &[Self::S]) -> Self::C;
    fn calc_postsplit(yl: &[Self::S], yr: &[Self::S]) -> Self::C;
}

/// Prediction of the final Leaf value.
pub trait LeafPredictor
{
    type Output;
    type S: Sample;

    /// predicted value
    fn predict(&self, x: &<Self::S as Sample>::X) -> Self::Output;

    /// fit predictor to data
    fn fit(data: &[Self::S]) -> Self;
}

/// The probabilistic leaf predictor models uncertainty in the prediction.
pub trait ProbabilisticLeafPredictor: LeafPredictor
{
    /// probability of given output `p(y|x)`
    fn prob(&self, s: &Self::S) -> Real;
}

/// Splits data at a tree node. This is a marker trait, shared by more specialized Splitters.
pub trait Splitter {
    type S: Sample;
    fn theta(&self) -> &<Self::S as Sample>::Theta;
}

/// Assigns a sample to either side of the split.
pub trait DeterministicSplitter: Splitter {
    //fn split(&self, f: &<Self::F as FeatureSet>::Sample::Output) -> Side;
    fn split(&self, x: &<Self::S as Sample>::X) -> Side;
}

/// Assigns a sample to both sides of the split with some probability each.
pub trait ProbabilisticSplitter: Splitter {
    /// Probability that the sample belongs to the left side of the split
    fn p_left(&self, x: &<Self::S as Sample>::X) -> Real;

    /// Probability that the sample belongs to the right side of the split
    fn p_right(&self, x: &<Self::S as Sample>::X) -> Real { 1.0 - self.p_left(x) }
}

/// Trait that allows a Splitter to generate random splits
pub trait RandomSplit<S: Splitter> {
    fn new_random<R: Rng>(data: &[S::S], rng: &mut R) -> Option<S>;
}

/// Find split
pub trait SplitFitter: Default {
    type S: Sample;
    type Split: Splitter<S=Self::S>;
    type Criterion: SplitCriterion<S=Self::S>;
    fn find_split(&self, data: &mut [Self::S]) -> Option<Self::Split>;
}

/// Trait that allows a type to be fitted
pub trait Learner<S: Sample, Output=Self>: Default {
    fn fit(&self, data: &[S]) -> Output;
}

/// Trait that allows a type to mutate the data set while being fitted
pub trait LearnerMut<S: Sample, Output=Self>: Default {
    fn fit(&self, data: &mut [S]) -> Output;
}

/// Trait that allows a type to predict values
pub trait Predictor<X, Y> {
    fn predict(&self, s: &X) -> Y;
}
