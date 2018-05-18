extern crate forester;
extern crate mldata;
extern crate ndarray;
extern crate rand;

mod common;

use std::fmt;

use ndarray::ArrayView2;
use rand::{thread_rng, Rng};

use forester::array_ops::Partition;
use forester::data::{SampleDescription, TrainingData};
use forester::dforest::DeterministicForestBuilder;
use forester::dtree::DeterministicTreeBuilder;
use forester::split::{BestRandomSplit, Split};
use forester::categorical::{Categorical, CatCount};

use mldata::mldata_mnist_original as mnist;

use common::dig_classes::{Digit, ClassCounts};

/// The mnist data set is pretty restrictive in terms of access. We can only get a sample given its
/// index. So we implement our Sample representation as a reference to the data set and an index.
///
/// Alternatively, we could have copied the data into our own structure.
struct Sample<'a> {
    data_set: &'a mnist::Data,
    index: usize,
}

impl<'a> fmt::Debug for Sample<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.data_set.get_sample(self.index))
    }
}

impl<'a> Sample<'a> {
    /// Convenience method to access the sample's features
    fn x(&self) -> ArrayView2<u8> {
        self.data_set.get_sample(self.index).0
    }

    /// Convenience method to access the sample's class label
    fn y(&self) -> Digit {
        Digit(self.data_set.get_sample(self.index).1)
    }
}

impl<'a> SampleDescription for Sample<'a> {
    type ThetaSplit = (usize, usize);
    type ThetaLeaf = ClassCounts;
    type Feature = u8;
    type Prediction = ClassCounts;

    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
        // We use the data columns directly as features
        self.x()[*theta]
    }

    fn sample_predict(&self, c: &Self::ThetaLeaf) -> Self::Prediction {
        c.clone()
    }
}

impl<'a> TrainingData<Sample<'a>> for [Sample<'a>] {
    fn n_samples(&self) -> usize {
        self.len()
    }

    fn gen_split_feature(&self) -> (usize, usize) {
        // Randomly select one of the 28x28 pixel
        (
            thread_rng().gen_range(0, 28),
            thread_rng().gen_range(0, 28),
        )
    }

    fn train_leaf_predictor(&self) -> ClassCounts {
        // count the number of samples in each class. This is possible
        // because there exists an `impl iter::Sum for ClassCounts`.
        self.iter().map(|sample| sample.y()).sum()
    }

    fn partition_data(&mut self, split: &Split<(usize, usize), u8>) -> (&mut Self, &mut Self) {
        // partition the data set over the split
        let i = self.partition(|sample| sample.sample_as_split_feature(&split.theta) <= split.threshold);
        // return two disjoint subsets
        self.split_at_mut(i)
    }

    fn split_criterion(&self) -> f64 {
        // This is a classification task, so we use the gini criterion.
        // In the future there will be a function provided by the library for this.
        let counts: ClassCounts = self.iter().map(|sample| sample.y()).sum();
        let gini = (0..10)
            .map(|c| counts.probability(Digit(c)))
            .map(|p| p * (1.0 - p))
            .sum();
        gini
    }

    fn feature_bounds(&self, theta: &(usize, usize)) -> (u8, u8) {
        // find minimum and maximum of a feature
        self.iter()
            .map(|sample| sample.sample_as_split_feature(theta))
            .fold((255, 0),
                  |(min, max), x| {
                      (if x < min {x} else {min},
                       if x > max {x} else {max})
                  })
    }
}


fn main() {
    // load data set
    let loader = mnist::DataSet::new().create().unwrap();
    let data = loader.load_data().unwrap();

    // convert data set for use with forester
    let mut training = Vec::new();
    let mut testing = Vec::new();
    for i in 0..data.n_samples() {
        let sample = Sample {
            data_set: &data,
            index: i,
        };

        // randomly assign samples to training and testing set
        if thread_rng().gen_range(0, 2) == 0 {
            testing.push(sample);
        } else {
            training.push(sample);
        }
    }

    println!("Fitting...");
    let forest = DeterministicForestBuilder::new(
        10,
        DeterministicTreeBuilder::new(
            10,
            None,
            BestRandomSplit::new(100)
        )
    ).fit(&mut training as &mut [_]);

    let mut confusion = [[0usize; 10]; 10];
    let mut totals = [0usize; 10];
    let mut n_correct = 0.0;

    println!("Predicting...");
    for sample in &testing {
        let probs = forest.predict(sample);
        let pred: Digit = probs.most_frequent();

        confusion[sample.y().as_usize()][pred.as_usize()] += 1;
        totals[sample.y().as_usize()] += 1;

        if sample.y() == pred {
            n_correct += 1.0;
        }
    }

    println!("Confusion matrix:");
    for r in 0..10 {
        print!("{} : ", r);
        for c in 0..10 {
            print!(" {:.2} ", confusion[r][c] as f64 / totals[r] as f64)
        }
        println!();
    }

    println!("\n Accuracy: {} / {} = {}", n_correct, testing.len(), n_correct / testing.len() as f64)
}
