extern crate examples_common;
extern crate forester;
extern crate openml;
extern crate rand;
#[macro_use]
extern crate serde_derive;

use rand::{thread_rng, Rng};
use openml::MeasureAccumulator;

use forester::data::{SampleDescription, TrainingData};
use forester::dforest::DeterministicForestBuilder;
use forester::dtree::DeterministicTreeBuilder;
use forester::split::BestRandomSplit;
use forester::categorical::CatCount;

use examples_common::dig_classes::ClassCounts;

#[derive(Deserialize)]
struct Features(Vec<u8>);

#[derive(Clone)]
struct Sample<'a> {
    x: &'a Features,
    y: u8,
}

impl<'a> From<(&'a Features, &'a u8)> for Sample<'a> {
    fn from(src: (&'a Features, &'a u8)) -> Self {
        Sample {
            x: src.0,
            y: *src.1,
        }
    }
}

impl<'a> SampleDescription for Sample<'a> {
    type ThetaSplit = usize;
    type ThetaLeaf = ClassCounts;
    type Feature = u8;
    type Prediction = ClassCounts;

    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
        // We use the data columns directly as features
        self.x.0[*theta]
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

    fn feature_bounds(&self, theta: &usize) -> (u8, u8) {
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
    let task = openml::SupervisedClassification::from_openml(146825).unwrap();

    println!("Task: {}", task.name());

    let acc: openml::PredictiveAccuracy<_> = task.run_static(|train, test| {

        let mut train: Vec<_> = train.map(Sample::from).collect();

        println!("Fitting...");
        let forest = DeterministicForestBuilder::new(
            100,
            DeterministicTreeBuilder::new(
                1,
                BestRandomSplit::new(10)
            ).with_bootstrap(1000)
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
    println!("{:#?}", acc);
    println!("{:#?}", acc.result());
}
