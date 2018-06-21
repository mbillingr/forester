extern crate examples_common;
extern crate forester;
extern crate num_traits;
extern crate openml;
extern crate rand;

use std::f32;
use std::fmt;

use rand::{thread_rng, Rng};
use openml::MeasureAccumulator;

use forester::array_ops::Partition;
use forester::data::{SampleDescription, TrainingData};
use forester::dforest::DeterministicForestBuilder;
use forester::dtree::DeterministicTreeBuilder;
use forester::iter_mean::IterMean;
use forester::split::{BestRandomSplit, Split};

struct Sample<'a> {
    x: &'a [Option<f32>],
    y: f32,
}

impl<'a> fmt::Debug for Sample<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} : {:?}", self.x, self.y)
    }
}

impl<'a> SampleDescription for Sample<'a> {
    type ThetaSplit = usize;
    type ThetaLeaf = f32;
    type Feature = f32;
    type Prediction = f32;

    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
        // We use the data columns directly as features
        match self.x[*theta] {
            Some(x) => x,
            None => -1.0,  // encode missing values as -1.0... this is pretty naive :)
        }
    }

    fn sample_predict(&self, c: &Self::ThetaLeaf) -> Self::Prediction {
        *c
    }
}

impl<'a> TrainingData<Sample<'a>> for [Sample<'a>] {
    fn n_samples(&self) -> usize {
        self.len()
    }

    fn gen_split_feature(&self) -> usize {
        // The data set has four feature columns
        thread_rng().gen_range(0, 4)
    }

    fn train_leaf_predictor(&self) -> f32 {
        // leaf prediction is the mean of all training samples that fall into the leaf
        f32::mean(self.iter().map(|sample| &sample.y))
    }

    fn partition_data(&mut self, split: &Split<usize, f32>) -> (&mut Self, &mut Self) {
        // partition the data set over the split
        let i = self.partition(|sample| sample.sample_as_split_feature(&split.theta) <= split.threshold);
        // return two disjoint subsets
        self.split_at_mut(i)
    }

    fn split_criterion(&self) -> f64 {
        // we use the variance in the target variable as splitting criterion
        let mean = f32::mean(self.iter().map(|sample| &sample.y));
        let variance = self.iter().map(|sample| sample.y - mean).map(|ym| ym * ym).sum::<f32>() / self.len() as f32;
        variance as f64
    }

    fn feature_bounds(&self, theta: &usize) -> (f32, f32) {
        // find minimum and maximum of a feature
        self.iter()
            .map(|sample| sample.sample_as_split_feature(theta))
            .fold((std::f32::INFINITY, std::f32::NEG_INFINITY),
                  |(min, max), x| {
                      (if x < min {x} else {min},
                       if x > max {x} else {max})
                  })
    }
}


fn main() {
    let task = openml::SupervisedRegression::from_openml(2285).unwrap();

    println!("Task: {}", task.name());

    // Fit a naive model as baseline: simply use the mean of the training data to predict the output
    let rmse0: openml::RootMeanSquaredError<_> = task.run(|train, _test| {
        let y_mean = train
            .map(|(_, y): (&[Option<f32>], &f32)| y)
            .fold((0, 0.0), |(mut n, mut mean), y| {
                n += 1usize;
                mean += (y - mean) / n as f32;
                (n, mean)
            })
            .1;

        Box::new(std::iter::repeat(y_mean))
    });

    // Now fit a random forest model
    let rmse: openml::RootMeanSquaredError<_> = task.run(|train, test| {

        let mut train: Vec<_> = train.map(|(x, &y)| Sample {x, y}).collect();

        println!("Fitting...");
        let forest = DeterministicForestBuilder::new(
            1000,
            DeterministicTreeBuilder::new(
                30,
                None,
                BestRandomSplit::new(3)
            )
        ).fit(&mut train as &mut [_]);

        println!("Predicting...");
        let result: Vec<_> = test.map(|x| {
            let sample = Sample {
                x,
                y: f32::NAN,
            };
            forest.predict(&sample)
        }).collect();

        Box::new(result.into_iter())
    });

    // compare the prediction performance of both models
    println!("\nNaive Model:");
    println!("RMSE: {:#?}", rmse0.result());

    println!("\nForest Regressor:");
    println!("RMSE: {:#?}", rmse.result());
}
