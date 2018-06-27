extern crate examples_common;
extern crate forester;
extern crate openml;
extern crate rand;

use std::fmt;

use rand::{thread_rng, Rng};

use forester::data::{SampleDescription, TrainingData};
use forester::categorical::CatCount;

use examples_common::dig_classes::{Digit, ClassCounts};

#[derive(Clone)]
struct Sample<'a> {
    x: &'a [u8],
    y: Digit,
}

impl<'a> fmt::Debug for Sample<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} : {:?}", self.x, self.y)
    }
}

impl<'a> From<(&'a [u8], &'a Digit)> for Sample<'a> {
    fn from(src: (&'a [u8], &'a Digit)) -> Self {
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
        thread_rng().gen_range(0, 28*28)
    }

    fn train_leaf_predictor(&self) -> ClassCounts {
        // count the number of samples in each class. This is possible
        // because there exists an `impl iter::Sum for ClassCounts`.
        self.iter().map(|sample| sample.y).sum()
    }

    fn split_criterion(&self) -> f64 {
        // This is a classification task, so we use the gini criterion.
        // In the future there will be a function provided by the library for this.
        let counts: ClassCounts = self
            .iter()
            .map(|sample| sample.y)
            .sum();

        let gini = (0..10)
            .map(|c| counts.probability(c))
            .map(|p| p * (1.0 - p))
            .sum();

        gini
    }

    fn feature_bounds(&self, _theta: &usize) -> (u8, u8) {
        // find minimum and maximum of a feature
        /*self.iter()
            .map(|sample| sample.sample_as_split_feature(theta))
            .fold((255, 0),
                  |(min, max), x| {
                      (if x < min {x} else {min},
                       if x > max {x} else {max})
                  })*/
        (0, 255)
    }
}

pub fn main() {
    #[cfg(feature = "cpuprofiler")] {
        extern crate cpuprofiler;

        use openml::MeasureAccumulator;
        use forester::dforest::DeterministicForestBuilder;
        use forester::dtree::DeterministicTreeBuilder;
        use forester::split::BestRandomSplit;

        let task = openml::SupervisedClassification::from_openml(146825).unwrap();
        println!("Task: {}", task.name());

        cpuprofiler::PROFILER.lock().unwrap().start("task.profile").unwrap();

        let acc: openml::PredictiveAccuracy<_> = task.run(|train, test| {

            let mut train: Vec<_> = train.map(Sample::from).collect();

            println!("Fitting...");
            let forest = DeterministicForestBuilder::new(
                100,
                DeterministicTreeBuilder::new(
                    1000,
                    None,
                    BestRandomSplit::new(10)
                )
            ).fit(&mut train as &mut [_]);

            println!("Predicting...");
            let result: Vec<_> = test
                .map(|x| {
                    let sample = Sample {
                        x,
                        y: Digit(99)
                    };
                    let prediction: Digit = forest.predict(&sample).most_frequent();
                    prediction
                })
                .collect();

            Box::new(result.into_iter())
        });

        cpuprofiler::PROFILER.lock().unwrap().stop().unwrap();

        println!("{:#?}", acc);
        println!("{:#?}", acc.result());

        println!("Profiling done. Convert the profile with something like");
        println!("  > pprof --callgrind target/release/examples/profile2 task.profile > task.prof");
        println!("Or view it with\n  > pprof --gv target/release/examples/profile2 task.profile");
    }
}
