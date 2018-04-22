//! Module for describing data.
//!
//! This module defines the traits required to define data sets for use with the forester crate.

use rand::distributions::range::SampleRange;

use split::Split;

/// Data sample used with decision trees
pub trait SampleDescription {
    /// Type used to parametrize split features
    type ThetaSplit;

    /// Type used to parametrize leaf predictors
    type ThetaLeaf;

    /// Type of a split feature
    type Feature: PartialOrd + SampleRange;

    /// Type of predicted values; this can be the same as `Self::Y` (e.g. regression) or something
    /// different (e.g. class probabilities).
    type Prediction;

    /// Compute the value of a leaf feature for a given sample
    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature;

    /// Compute the leaf prediction for a given sample
    fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction;
}

/// Data set that can be used for training decision trees
pub trait TrainingData<Sample>
    where Sample: SampleDescription
{
    /// Return number of samples in the data set
    fn n_samples(&self) -> usize;

    /// Generate a new split feature (typically, this will be randomized)
    fn gen_split_feature(&self) -> Sample::ThetaSplit;

    /// Train a new leaf predictor
    fn train_leaf_predictor(&self) -> Sample::ThetaLeaf;

    /// Partition data set in-place according to a split
    fn partition_data(&mut self, split: &Split<Sample::ThetaSplit, Sample::Feature>) -> (&mut Self, &mut Self);

    /// Compute split criterion
    fn split_criterion(&self) -> f64;

    /// Return minimum and maximum value of a feature
    fn feature_bounds(&self, theta: &Sample::ThetaSplit) -> (Sample::Feature, Sample::Feature);
}
