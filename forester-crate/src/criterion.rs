use std::f64;

use categorical::{Categorical, CatCount, GenericCatCounter};
use continuous::Continuous;
use data::{SampleDescription, TrainingData};

/// Criterion for evaluating splits
///
/// Criteria should be positive values, where larger values correspond to worst splits (such as
/// GINI or variance). A criterion of exactly 0.0 indicates a perfect split and may be used by the
/// split finder to stop early.
pub trait SplitCriterion<T>
    where Self: Sized
{
    /// Initialize criterion
    fn new() -> Self;

    /// Add a sample
    fn add_sample<S: SampleDescription<Target=T>>(&mut self, sample: &S);

    /// Remove a sample
    fn remove_sample<S: SampleDescription<Target=T>>(&mut self, sample: &S);

    /// Compute final value
    fn get(&self) -> f64;

    /// Compute final value weighted by number of samples
    ///
    /// This is equivalent to `.get() * n_samples`.
    fn get_weighted(&self) -> f64;

    /// Construct the criterion from a data set
    fn from_dataset<S, D>(data: &D) -> Self
        where S: SampleDescription<Target=T>,
              D: ?Sized + TrainingData<S, Criterion=Self>
    {
        let mut crit = Self::new();
        data.visit_samples(|s| crit.add_sample(s));
        crit
    }
}

/// Variance criterion for evaluating splits in regression tasks
pub struct VarianceCriterion {
    n: usize,
    mean: f64,
    m2: f64,
}

impl<T> SplitCriterion<T> for VarianceCriterion
where T: Continuous
{
    fn new() -> Self {
        VarianceCriterion {
            n: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    fn add_sample<S: SampleDescription<Target=T>>(&mut self, sample: &S) {
        let x = sample.target().as_float();
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }

    fn remove_sample<S: SampleDescription<Target=T>>(&mut self, sample: &S) {
        debug_assert!(self.n > 0);
        let x = sample.target().as_float();
        let delta = x - self.mean;
        self.mean = (self.n as f64 * self.mean - x) / (self.n - 1) as f64;
        self.m2 -= delta * (x - self.mean);
        self.n -= 1;
    }

    fn get(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.m2 / self.n as f64
        }
    }

    fn get_weighted(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.m2
        }
    }
}

/// GINI criterion for evaluating splits in a classification task with an
/// arbitrary number of classes.
pub struct GiniCriterion {
    counts: GenericCatCounter,
    n: usize,
}

impl<T> SplitCriterion<T> for GiniCriterion
where T: Categorical
{
    fn new() -> Self {
        GiniCriterion {
            counts: GenericCatCounter::new(),
            n: 0,
        }
    }

    fn add_sample<S: SampleDescription<Target=T>>(&mut self, sample: &S) {
        self.counts.add(sample.target());
        self.n += 1;
    }

    fn remove_sample<S: SampleDescription<Target=T>>(&mut self, sample: &S) {
        self.counts.remove(sample.target());
        self.n += 1;
    }

    fn get(&self) -> f64 {
        let mut gini = 0.0;
        self.counts.probs(|p| gini += p * (1.0 - p));
        gini
    }

    fn get_weighted(&self) -> f64 {
        let mut gini = 0.0;
        self.counts.probs(|p| gini += p * (1.0 - p));
        gini * self.n as f64
    }
}
