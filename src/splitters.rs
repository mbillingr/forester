use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;

use rand::distributions::range::SampleRange;
use rand::Rng;

use super::DataSet;
use super::DeterministicSplitter;
use super::Feature;
use super::RandomSplit;
use super::Sample;
use super::Side;
use super::SplitCriterion;
use super::SplitFitter;
use super::Splitter;

use random::DefaultRng;

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
    fn new_random<R: Rng>(data: &D, rng: &mut R) -> Option<ThresholdSplitter<D>> {
        let theta = data.random_feature(rng);

        let f = D::FX::get_feature(&data.get(0).get_x(), &theta);

        let (min, max) = data.reduce_feature(&theta, (f.clone(), f),
                                             |minmax, f| (
                                                 if f < minmax.0 { f.clone() } else { minmax.0 },
                                                 if f > minmax.1 { f } else { minmax.1 }
                                             )
        );

        if min >= max {
            return None
        }

        let threshold = rng.gen_range(min, max);

        Some(ThresholdSplitter {
            theta,
            threshold,
        })
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

pub struct BestRandomSplit<S, C, R: Rng> {
    n_splits: usize,
    rng: RefCell<R>,
    _p: PhantomData<(S, C)>,
}

impl<S, C, R: DefaultRng> BestRandomSplit<S, C, R> {
    pub fn new(n_splits: usize, rng: R) -> Self {
        BestRandomSplit {
            n_splits,
            rng: RefCell::new(rng),
            _p: PhantomData,
        }
    }
}

impl<S, C, R: DefaultRng> Default for BestRandomSplit<S, C, R> {
    fn default() -> Self {
        BestRandomSplit {
            n_splits: 10,
            rng: RefCell::new(R::default_rng()),
            _p: PhantomData,
        }
    }
}

// this specialized for float 64 criteria... this is not really necessary, but allows an optimization
impl<S: DeterministicSplitter + RandomSplit<S>, C: SplitCriterion<D=S::D, C=f64>, R: DefaultRng> SplitFitter for BestRandomSplit<S, C, R>
{
    type D = S::D;
    type Split = S;
    type Criterion = C;
    fn find_split(&self, data: &mut Self::D) -> Option<Self::Split> {
        let mut best_criterion = None;
        let mut best_split = None;

        let mut rng = self.rng.borrow_mut();

        let parent_criterion = C::calc_presplit(data);

        for _ in 0..self.n_splits {
            let split: S = match Self::Split::new_random(data, &mut *rng) {
                Some(s) => s,
                None => continue
            };

            let i = data.partition_by_split(&split);

            let (left, right) = data.subsets(i);

            let new_criterion = C::calc_postsplit(left, right);

            let swap = match best_criterion {
                None => new_criterion <= parent_criterion,
                Some(c) => new_criterion < c,
            };

            if swap {
                best_criterion = Some(new_criterion);
                best_split = Some(split);
            }

            // stop early if we find a perfect split
            if let Some(c) = best_criterion {
                if c == 0.0 {
                    break
                }
            }
        }

        best_split
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    use rand::thread_rng;

    use super::super::SplitFitter;
    use criteria::VarCriterion;
    use datasets::TupleSample;
    use features::ColumnSelect;

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

        let brs: BestRandomSplit<ThresholdSplitter<[Sample]>, VarCriterion<_>, _> = BestRandomSplit {
            n_splits: 100,  // Make *almost* sure that we will find the optimal split
            rng: RefCell::new(thread_rng()),
            _p: PhantomData
        };

        let split = brs.find_split(data).expect("No split found");
        assert_eq!(split.theta, 0);
        assert_eq!(split.threshold, 3);
    }

    #[test]
    fn best_random_constfeature() {
        type Sample = TupleSample<ColumnSelect, [f32;1], f64>;

        let data: &mut [Sample] = &mut [
            TupleSample::new([42.0], 1.0),
            TupleSample::new([42.0], 2.0),
            TupleSample::new([42.0], 1.0),
            TupleSample::new([42.0], 2.0),
            TupleSample::new([42.0], 11.0),
            TupleSample::new([42.0], 12.0),
            TupleSample::new([42.0], 11.0),
            TupleSample::new([42.0], 12.0),
        ];

        let brs: BestRandomSplit<ThresholdSplitter<[Sample]>, VarCriterion<_>, _> = BestRandomSplit {
            n_splits: 100,  // Make *almost* sure that we will find the optimal split
            rng: RefCell::new(thread_rng()),
            _p: PhantomData
        };

        let split = brs.find_split(data);
        assert_eq!(split.is_none(), true);
    }
}
