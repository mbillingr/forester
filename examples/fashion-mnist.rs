extern crate forester;
extern crate openml;
extern crate rand;

mod common;

use std::fmt;

use rand::{thread_rng, Rng};

use forester::array_ops::Partition;
use forester::data::{SampleDescription, TrainingData};
use forester::dforest::DeterministicForestBuilder;
use forester::dtree::DeterministicTreeBuilder;
use forester::split::{BestRandomSplit, Split};
use forester::categorical::{Categorical, CatCount};

use openml::OpenML;

use common::dig_classes::ClassCounts;

struct Sample<'a> {
    x: &'a [f64],
    y: u8,
}

impl<'a> fmt::Debug for Sample<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} : {:?}", self.x, self.y)
    }
}

impl<'a> SampleDescription for Sample<'a> {
    type ThetaSplit = usize;
    type ThetaLeaf = ClassCounts;
    type Feature = f64;
    type Prediction = ClassCounts;

    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
        // We use the data columns directly as features
        self.x[*theta]
    }

    fn sample_predict(&self, c: &Self::ThetaLeaf) -> Self::Prediction {
        c.clone()
    }
}

impl<'a> TrainingData<Sample<'a>> for [Sample<'a>] {
    fn n_samples(&self) -> usize {
        self.len()
    }

    fn gen_split_feature(&self) -> usize {
        // The data set has four feature columns
        thread_rng().gen_range(0, 784)
    }

    fn train_leaf_predictor(&self) -> ClassCounts {
        // count the number of samples in each class. This is possible
        // because there exists an `impl iter::Sum for ClassCounts`.
        self.iter().map(|sample| sample.y).sum()
    }

    fn partition_data(&mut self, split: &Split<usize, f64>) -> (&mut Self, &mut Self) {
        // partition the data set over the split
        let i = self.partition(|sample| sample.sample_as_split_feature(&split.theta) <= split.threshold);
        // return two disjoint subsets
        self.split_at_mut(i)
    }

    fn split_criterion(&self) -> f64 {
        // This is a classification task, so we use the gini criterion.
        // In the future there will be a function provided by the library for this.
        let counts: ClassCounts = self.iter().map(|sample| sample.y).sum();

        let mut gini = 0.0;
        for c in 0..10 {
            let p = counts.probability(c);
            gini += p * (1.0 - p);
        }
        gini
    }

    fn feature_bounds(&self, theta: &usize) -> (f64, f64) {
        // find minimum and maximum of a feature
        self.iter()
            .map(|sample| sample.sample_as_split_feature(theta))
            .fold((std::f64::INFINITY, std::f64::NEG_INFINITY),
                  |(min, max), x| {
                      (if x < min {x} else {min},
                       if x > max {x} else {max})
                  })
    }
}


fn main() {
    let task = OpenML::new().task(146825).unwrap();

    println!("Task: {}", task.name());

    let measure = task.run(|train, test| {

        let mut train: Vec<_> = train.map(|(x, &y)| Sample {x, y}).collect();

        println!("Fitting...");
        let forest = DeterministicForestBuilder::new(
            100,
            DeterministicTreeBuilder::new(
                1000,
                None,
                BestRandomSplit::new(1)
            )
        ).fit(&mut train as &mut [_]);

        println!("Predicting...");
        let result: Vec<_> = test
            .map(|x| {
                let sample = Sample {
                    x,
                    y: 99
                };
                let prediction: u8 = forest.predict(&sample).most_frequent();
                prediction
            })
            .collect();

        Box::new(result.into_iter())
    });
    println!("{:#?}", measure);
    println!("{:#?}", measure.result());
}
