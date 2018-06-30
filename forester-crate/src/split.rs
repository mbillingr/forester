//! Module for working with splits.

use std::f64;

use rand::{thread_rng, Rng};

use criterion::SplitCriterion;
use data::{SampleDescription, TrainingData};
use split_between::SplitBetween;

/// Parametric representation of a split.
///
/// A split consists of a data-set dependent parameter `theta` that corresponds to a feature, and
/// a threshold to split the feature into two half-spaces.
#[derive(Debug)]
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

/// Find best split, in a number of randomly selected features.
///
/// Normally, exactly `n_features` are tested. However, there are two notable exceptions:
/// 1. if a "perfect split" is encountered (split criterion is exactly zero) the search stops early.
/// 2. if after no split has been found after `n_features` the search continues until it finds one
///    or runs out of features.
pub struct BestSplitRandomFeature {
    n_features: usize,
}

impl BestSplitRandomFeature {
    pub fn new(n_features: usize) -> Self {
        BestSplitRandomFeature {
            n_features
        }
    }
}

impl SplitFinder for BestSplitRandomFeature
{
    fn find_split<Sample, Training>(&self, data: &mut Training)
                                    -> Option<Split<Sample::ThetaSplit, Sample::Feature>>
        where Sample: SampleDescription,
              Training: ?Sized + TrainingData<Sample>
    {
        let mut best_criterion = Training::Criterion::from_dataset(data).get();
        let mut best_split = None;

        let mut features: Vec<_> = data
            .all_split_features()
            .expect("Dataset does not support iteration over features.")
            .collect();

        thread_rng().shuffle(&mut features);

        let mut n_to_check = self.n_features;

        for theta in features {
            if best_criterion == 0.0 {
                break
            }

            if n_to_check == 0 {
                if best_split.is_some() {
                    break
                }
            } else {
                n_to_check -= 1;
            }

            let (criterion, split) = find_best_split_for_feature(data, &theta);

            if criterion <= best_criterion {
                best_criterion = criterion;
                best_split = split;
            }

        }

        best_split
    }
}

/// Find best split.
///
/// Finds best split among all possible splits in all features. This only works for data sets that
/// implement `all_split_features` to return not `None`.
pub struct BestSplit {
}

impl BestSplit {
    pub fn new() -> Self {
        BestSplit {
        }
    }
}

impl SplitFinder for BestSplit
{
    fn find_split<Sample, Training>(&self, data: &mut Training)
                                    -> Option<Split<Sample::ThetaSplit, Sample::Feature>>
        where Sample: SampleDescription,
              Training: ?Sized + TrainingData<Sample>
    {
        let mut best_criterion = Training::Criterion::from_dataset(data).get();
        let mut best_split = None;

        let features = data.all_split_features().expect("Dataset does not support iteration over features.");

        for theta in features {
            if best_criterion == 0.0 {
                break
            }

            let (criterion, split) = find_best_split_for_feature(data, &theta);

            if criterion <= best_criterion {
                best_criterion = criterion;
                best_split = split;
            }
        }

        best_split
    }
}

/// find optimal split for given feature
fn find_best_split_for_feature<Sample, Training>(data: &mut Training, theta: &Sample::ThetaSplit)
                                                 -> (f64, Option<Split<Sample::ThetaSplit, Sample::Feature>>)
    where Sample: SampleDescription,
          Training: ?Sized + TrainingData<Sample>
{
    data.sort_data(theta);

    let mut best_criterion = f64::INFINITY;
    let mut best_split = None;

    let mut left_crit = Training::Criterion::from_dataset(data);
    let mut right_crit = Training::Criterion::new();

    let mut prev_sf: Option<Sample::Feature> = None;
    data.visit_samples(|sample| {
        let sf = sample.sample_as_split_feature(theta);

        // don't try to split between two samples of same feature value
        if let Some(ref psf) = prev_sf {
            if psf == &sf {
                left_crit.remove_sample(sample);
                right_crit.add_sample(sample);
                return
            }
        }

        let criterion = (left_crit.get_weighted() + right_crit.get_weighted()) / data.n_samples() as f64;

        if criterion <= best_criterion {
            if let Some(ref psf) = prev_sf {
                best_criterion = criterion;
                best_split = Some(Split {
                    theta: theta.clone(),
                    threshold: psf.split_between(&sf)
                });
            }
        }

        left_crit.remove_sample(sample);
        right_crit.add_sample(sample);

        prev_sf = Some(sf);
    });

    (best_criterion, best_split)
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

    #[test]
    fn best_split() {
        let data: &mut [_] = &mut [
            Sample::new(&[0.0, 41.0, 0.0], 1.0),
            Sample::new(&[0.0, 41.0, 1.0], 2.0),
            Sample::new(&[0.0, 43.0, 2.0], 1.0),
            Sample::new(&[0.0, 43.0, 3.0], 2.0),
            Sample::new(&[0.0, 41.0, 4.0], 11.0),
            Sample::new(&[0.0, 41.0, 5.0], 12.0),
            Sample::new(&[0.0, 43.0, 6.0], 11.0),
            Sample::new(&[0.0, 43.0, 7.0], 12.0),
            Sample::new(&[0.0, 42.0, 8.0], 11.0),
            Sample::new(&[0.0, 42.0, 9.0], 12.0),
        ];
        let spl = BestSplit::new();
        let split = spl.find_split(data).unwrap();
        assert_eq!(split.theta, 2);
        assert!(split.threshold >= 3.0);
        assert!(split.threshold <= 4.0);
    }

    #[test]
    fn best_split_random_feature() {
        let data: &mut [_] = &mut [
            Sample::new(&[0.0, 41.0, 0.0], 1.0),
            Sample::new(&[0.0, 41.0, 1.0], 2.0),
            Sample::new(&[0.0, 43.0, 2.0], 1.0),
            Sample::new(&[0.0, 43.0, 3.0], 2.0),
            Sample::new(&[0.0, 41.0, 4.0], 11.0),
            Sample::new(&[0.0, 41.0, 5.0], 12.0),
            Sample::new(&[0.0, 43.0, 6.0], 11.0),
            Sample::new(&[0.0, 43.0, 7.0], 12.0),
            Sample::new(&[0.0, 42.0, 8.0], 11.0),
            Sample::new(&[0.0, 42.0, 9.0], 12.0),
        ];
        let spl = BestSplitRandomFeature::new(3);
        let split = spl.find_split(data).unwrap();
        assert_eq!(split.theta, 2);
        assert!(split.threshold >= 3.0);
        assert!(split.threshold <= 4.0);
    }
}
