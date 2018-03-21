use std::fmt;
use std::marker::PhantomData;

use rand::distributions::{IndependentSample, Range};
use rand::distributions::range::SampleRange;
use rand::{thread_rng, Rng};

use super::DataSet;
use super::DeterministicSplitter;
use super::Feature;
use super::RandomSplit;
use super::Sample;
use super::Side;
use super::SplitCriterion;
use super::SplitFitter;
use super::Splitter;

/// Use a simple threshold for deterministic split.
pub struct ThresholdSplitter<D>
    where D: ?Sized + DataSet,
{
    theta: D::Theta,
    threshold: D::F,
}

impl<D> ThresholdSplitter<D>
    where D: ?Sized + DataSet
{
    pub fn new(theta: D::Theta, threshold: D::F) -> Self {
        ThresholdSplitter {
            theta,
            threshold
        }
    }
}

impl<D> fmt::Debug for ThresholdSplitter<D>
    where D: ?Sized + DataSet,
          D::Theta: fmt::Debug,
          D::F: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ThresholdSplitter (theta: {:?}, threshold: {:?})", self.theta, self.threshold)
    }
}

impl<D> Splitter for ThresholdSplitter<D>
    where D: ?Sized + DataSet
{
    type D = D;

    fn theta(&self) -> &<Self::D as DataSet>::Theta {
        &self.theta
    }
}

impl<D> RandomSplit<ThresholdSplitter<D>> for ThresholdSplitter<D>
    where D: ?Sized + DataSet,
          D::F: SampleRange
{
    fn new_random(data: &D) -> ThresholdSplitter<D> {
        let theta = data.random_feature();

        let minmax = data.reduce_feature(&theta, None, |minmax, f|{
            match minmax {
                None => Some((f.clone(), f)),
                Some((mut min, mut max)) => {
                    if f < min { min = f.clone(); }
                    if f > max { max = f; }
                    Some((min, max))
                }
            }
        });
        let (min, max) = minmax.expect("Unable to determine feature range");

        let threshold = thread_rng().gen_range(min, max);

        ThresholdSplitter {
            theta,
            threshold,
        }
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

pub struct BestRandomSplit<S, C> {
    n_splits: usize,
    _p: PhantomData<(S, C)>,
}

impl<S, C> BestRandomSplit<S, C> {
    pub fn new(n_splits: usize) -> Self {
        BestRandomSplit {
            n_splits,
            _p: PhantomData,
        }
    }
}

impl<S, C> Default for BestRandomSplit<S, C> {
    fn default() -> Self {
        BestRandomSplit {
            n_splits: 10,
            _p: PhantomData,
        }
    }
}

impl<S: DeterministicSplitter + RandomSplit<S>, C: SplitCriterion<D=S::D>> SplitFitter for BestRandomSplit<S, C> {
    type D = S::D;
    type Split = S;
    type Criterion = C;
    fn find_split(&self, data: &mut Self::D) -> Option<Self::Split> {
        let mut best_criterion = None;
        let mut best_split = None;

        let parent_criterion = C::calc_presplit(data);

        for _ in 0..self.n_splits {
            let split: S = Self::Split::new_random(data);

            let i = data.partition_by_split(&split);

            let (left, right) = data.subsets(i);

            let new_criterion = C::calc_postsplit(left, right);

            let swap = match best_criterion {
                None => true,
                Some(c) => new_criterion < c,
            };

            if swap {
                best_criterion = Some(new_criterion);
                best_split = Some(split);
            }
        }

        best_split
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use super::super::SplitFitter;
    use criteria::VarCriterion;
    use datasets::TupleSample;
    use features::ColumnSelect;
    use predictors::ConstMean;

    #[test]
    fn best_random() {
        type Sample = TupleSample<ColumnSelect, [i32;1], f64>;

        let data: &mut [Sample] = &mut [
            TupleSample::new([0], 1.0),
            TupleSample::new([1], 2.0),
            TupleSample::new([2], 1.0),
            TupleSample::new([3], 2.0),
            TupleSample::new([4], 11.0),
            TupleSample::new([5], 12.0),
            TupleSample::new([6], 11.0),
            TupleSample::new([7], 12.0),
        ];

        let brs: BestRandomSplit<ThresholdSplitter<[Sample]>, VarCriterion<_>> = BestRandomSplit {
            n_splits: 100,  // Make *almost* sure that we will find the optimal split
            _p: PhantomData
        };

        let split = brs.find_split(data).expect("No split found");
        assert_eq!(split.theta, 0);
        assert_eq!(split.threshold, 3);
    }
}
