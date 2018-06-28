//! Module for working with splits.

use rand::{thread_rng, Rng};

use criterion::SplitCriterion;
use data::{SampleDescription, TrainingData};

/// Parametric representation of a split.
///
/// A split consists of a data-set dependent parameter `theta` that corresponds to a feature, and
/// a threshold to split the feature into two half-spaces.
pub struct Split<Theta, Threshold> {
    pub theta: Theta,
    pub threshold: Threshold,
}

/// Find split
pub trait SplitFinder
{
    /// Attempt to find a split for the given data set.
    fn find_split<Sample, Training>(&self, data: &mut Training)
        -> Option<Split<Sample::ThetaSplit, Sample::Feature>>
        where Sample: SampleDescription,
              Training: ?Sized + TrainingData<Sample>;
}

/// Find best random split.
///
/// This structure is used to find an optimal random split. It generates `n_splits` random splits
/// and selects the best one according to a data-dependent criterion. A higher number of splits
/// increases the chance of finding a more optimal split but reduces diversity and is
/// computationally more demanding.
pub struct BestRandomSplit {
    n_splits: usize,
}

impl BestRandomSplit {
    pub fn new(n_splits: usize) -> Self {
        BestRandomSplit {
            n_splits
        }
    }
}

impl SplitFinder for BestRandomSplit
{
    fn find_split<Sample, Training>(&self, data: &mut Training)
        -> Option<Split<Sample::ThetaSplit, Sample::Feature>>
        where Sample: SampleDescription,
              Training: ?Sized + TrainingData<Sample>
    {
        let n = data.n_samples() as f64;
        //let mut best_criterion = data.split_criterion();
        let mut best_criterion = Training::Criterion::from_dataset(data).get();
        let mut best_split = None;

        let mut rng = thread_rng();

        for _ in 0..self.n_splits {
            let theta = data.gen_split_feature();

            let (min, max) = data.feature_bounds(&theta);

            if max <= min {
                continue
            }

            let threshold = rng.gen_range(min, max);

            let split = Split{theta, threshold};
            let (left, right) = data.partition_data(&split);

            let left_crit = Training::Criterion::from_dataset(left).get_weighted();
            let right_crit = Training::Criterion::from_dataset(right).get_weighted();
            let criterion = (left_crit + right_crit) / n;

            if criterion <= best_criterion {
                best_criterion = criterion;
                best_split = Some(split);
            }

            // stop early if we find a perfect split
            // TODO: tolerance rather than exact comparison?
            if best_criterion == 0.0 {
                break
            }
        }

        best_split
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use split::BestRandomSplit;
    use testdata::Sample;

    #[test]
    fn best_random_split() {
        let data: &mut [_] = &mut [
            Sample::new(&[0.0], 1.0),
            Sample::new(&[1.0], 2.0),
        ];
        let spl = BestRandomSplit::new(1);
        let split = spl.find_split(data).unwrap();
        assert_eq!(split.theta, 0);
        assert!(split.threshold >= 0.0);
        assert!(split.threshold <= 1.0);

        let data: &mut [_] = &mut [
            Sample::new(&[41.0, 0.0], 1.0),
            Sample::new(&[41.0, 1.0], 2.0),
            Sample::new(&[43.0, 2.0], 1.0),
            Sample::new(&[43.0, 3.0], 2.0),
            Sample::new(&[41.0, 4.0], 11.0),
            Sample::new(&[41.0, 5.0], 12.0),
            Sample::new(&[43.0, 6.0], 11.0),
            Sample::new(&[43.0, 7.0], 12.0),
        ];
        let spl = BestRandomSplit::new(100);
        let split = spl.find_split(data).unwrap();
        assert_eq!(split.theta, 1);
        assert!(split.threshold >= 3.0);
        assert!(split.threshold <= 4.0);
    }
}
