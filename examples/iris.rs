extern crate forester;
extern crate num_traits;
extern crate openml;
extern crate rand;
#[macro_use]
extern crate serde_derive;

mod common;

use std::fmt;

use num_traits::ToPrimitive;
use rand::{thread_rng, Rng};
use openml::MeasureAccumulator;

use forester::array_ops::Partition;
use forester::data::{SampleDescription, TrainingData};
use forester::dforest::DeterministicForestBuilder;
use forester::dtree::DeterministicTreeBuilder;
use forester::split::{BestRandomSplit, Split};
use forester::categorical::{Categorical, CatCount};

use common::rgb_classes::ClassCounts;

#[derive(Debug, Copy, Clone, Deserialize, Eq, PartialEq)]
enum Iris {
    None,

    #[serde(rename = "Iris-setosa")]
    Setosa,

    #[serde(rename = "Iris-versicolor")]
    Versicolor,

    #[serde(rename = "Iris-virginica")]
    Virginica,
}

impl<'a> From<&'a f64> for Iris {
    fn from(f: &'a f64) -> Iris {
        Iris::from_usize(*f as usize)
    }
}

impl Categorical for Iris {
    fn as_usize(&self) -> usize {
        match *self {
            Iris::Setosa => 0,
            Iris::Versicolor => 1,
            Iris::Virginica => 2,
            Iris::None => panic!("invalid Iris")
        }
    }

    fn from_usize(id: usize) -> Self {
        match id as u8 {
            0 => Iris::Setosa,
            1 => Iris::Versicolor,
            2 => Iris::Virginica,
            _ => Iris::None
        }
    }

    fn n_categories(&self) -> Option<usize> {
        Some(3)
    }
}

impl ToPrimitive for Iris {
    fn to_i64(&self) -> Option<i64> {
        Some(self.as_usize() as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(self.as_usize() as u64)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct IrisData {
    sepallength: f32,
    sepalwidth: f32,
    petallength: f32,
    petalwidth: f32,
}

struct Sample<'a> {
    x: &'a IrisData,
    y: Iris,
}

impl<'a> fmt::Debug for Sample<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} : {:?}", self.x, self.y)
    }
}

impl<'a> SampleDescription for Sample<'a> {
    type ThetaSplit = usize;
    type ThetaLeaf = ClassCounts;
    type Feature = f32;
    type Prediction = ClassCounts;

    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
        // We use the data columns directly as features
        match *theta {
            0 => self.x.sepallength,
            1 => self.x.sepalwidth,
            2 => self.x.petallength,
            3 => self.x.petalwidth,
            _ => panic!("Invalid feature")
        }
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
        thread_rng().gen_range(0, 4)
    }

    fn train_leaf_predictor(&self) -> ClassCounts {
        // count the number of samples in each class. This is possible
        // because there exists an `impl iter::Sum for ClassCounts`.
        self.iter().map(|sample| sample.y).sum()
    }

    fn partition_data(&mut self, split: &Split<usize, f32>) -> (&mut Self, &mut Self) {
        // partition the data set over the split
        let i = self.partition(|sample| sample.sample_as_split_feature(&split.theta) <= split.threshold);
        // return two disjoint subsets
        self.split_at_mut(i)
    }

    fn split_criterion(&self) -> f64 {
        // This is a classification task, so we use the gini criterion.
        // In the future there will be a function provided by the library for this.
        let counts: ClassCounts = self.iter().map(|sample| sample.y).sum();
        let p1 = counts.probability(Iris::Setosa);
        let p2 = counts.probability(Iris::Versicolor);
        let p3 = counts.probability(Iris::Virginica);
        let gini = p1 * (1.0 - p1) + p2 * (1.0 - p2) + p3 * (1.0 - p3);
        gini
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
    let task = openml::SupervisedClassification::from_openml(59).unwrap();

    println!("Task: {}", task.name());

    let acc: openml::PredictiveAccuracy<_> = task.run_static(|train, test| {

        let mut train: Vec<_> = train.map(|(x, &y)| Sample {x, y}).collect();

        println!("Fitting...");
        let forest = DeterministicForestBuilder::new(
            100,
            DeterministicTreeBuilder::new(
                15,
                None,
                BestRandomSplit::new(4)
            )
        ).fit(&mut train as &mut [_]);

        println!("Predicting...");
        let result: Vec<_> = test.map(|x| {
            let sample = Sample {
                x,
                y: Iris::None
            };
            let prediction: Iris = forest.predict(&sample).most_frequent();
            prediction
        }).collect();

        Box::new(result.into_iter())
    });
    println!("{:#?}", acc);
    println!("{:#?}", acc.result());
}
