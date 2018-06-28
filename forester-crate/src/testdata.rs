
use num_traits::Bounded;
use rand::distributions::range::SampleRange;
use rand::{thread_rng, Rng};

use criterion::VarianceCriterion;
use data::{SampleDescription, TrainingData};
use iter_mean::IterMean;

#[derive(Debug, Clone)]
pub struct Sample<'a, X: 'a, Y> {
    pub x: &'a[X],
    pub y: Y,
}

impl<'a, X: 'a, Y> Sample<'a, X, Y> {
    pub fn new(x: &'a[X], y: Y) -> Self {
        Sample {
            x,
            y,
        }
    }
}

impl<'a, X> SampleDescription for Sample<'a, X, f64>
    where X: Clone + PartialOrd + SampleRange + Bounded
{
    type ThetaSplit = usize;
    type ThetaLeaf = f64;
    type Feature = X;
    type Target = f64;
    type Prediction = f64;

    fn target(&self) -> Self::Target {
        self.y
    }

    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
        self.x[*theta].clone()
    }

    fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction {
        *w
    }
}

impl<'a, X> TrainingData<Sample<'a, X, f64>> for [Sample<'a, X, f64>]
    where X: Clone + PartialOrd + SampleRange + Bounded
{
    type Criterion = VarianceCriterion;

    fn n_samples(&self) -> usize {
        self.len()
    }

    fn gen_split_feature(&self) -> usize {
        thread_rng().gen_range(0, self[0].x.len())
    }

    fn train_leaf_predictor(&self) -> f64 {
        f64::mean(self.iter().map(|sample| &sample.y))
    }

    fn feature_bounds(&self, theta: &usize) -> (X, X) {
        self.iter()
            .map(|sample| sample.sample_as_split_feature(theta))
            .fold((X::max_value(), X::min_value()),
                  |(min, max), x| {
                      (if x < min {x.clone()} else {min},
                       if x > max {x} else {max})
                  })
    }
}