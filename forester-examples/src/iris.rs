extern crate examples_common;
extern crate forester;
extern crate num_traits;
extern crate openml;
extern crate rand;
#[macro_use]
extern crate serde_derive;

use std::fmt;

use num_traits::ToPrimitive;
use rand::{thread_rng, Rng};
use openml::MeasureAccumulator;

use forester::categorical::{Categorical, CatCount};
use forester::criterion::GiniCriterion;
use forester::data::{SampleDescription, TrainingData};
use forester::dforest::DeterministicForestBuilder;
use forester::dtree::DeterministicTreeBuilder;
use forester::split::BestRandomSplit;

use examples_common::rgb_classes::ClassCounts;

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

#[derive(Clone)]
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
    type Target = Iris;
    type Prediction = ClassCounts;

    fn target(&self) -> Self::Target {
        self.y
    }

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
    type Criterion = GiniCriterion; // TODO: specialized Gini implementation for three classes

    fn n_samples(&self) -> usize {
        self.len()
    }

    fn gen_split_feature(&self) -> usize {
        // The data set has four feature columns
        thread_rng().gen_range(0, 4)
    }

    fn all_split_features(&self) -> Option<Box<Iterator<Item=usize>>> {
        Some(Box::new(0..4))
    }

    fn train_leaf_predictor(&self) -> ClassCounts {
        // count the number of samples in each class. This is possible
        // because there exists an `impl iter::Sum for ClassCounts`.
        self.iter().map(|sample| sample.y).sum()
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
