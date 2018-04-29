extern crate forester;
extern crate mldata;
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

use mldata::uci_iris;

use common::rgb_classes::ClassCounts;

/// Wrap the data set's label type so that we can implement the `Categorical` trait for it.
#[derive(Debug, Copy, Clone)]
struct Iris(uci_iris::Iris);

/// The iris set is pretty restrictive in terms of access. We can only get a sample given its
/// index. So we implement our Sample representation as a reference to the data set and an index.
///
/// Alternatively, we could have copied the data into our own structure.
struct Sample<'a> {
    data_set: &'a uci_iris::Data,
    index: usize,
}

impl<'a> fmt::Debug for Sample<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.data_set.get_sample(self.index))
    }
}

impl<'a> Sample<'a> {
    /// Convenience method to access the sample's features
    fn x(&self) -> &[f32] {
        self.data_set.get_sample(self.index).0
    }

    /// Convenience method to access the sample's class label
    fn y(&self) -> Iris {
        Iris(self.data_set.get_sample(self.index).1)
    }
}

impl<'a> SampleDescription for Sample<'a> {
    type ThetaSplit = usize;
    type ThetaLeaf = ClassCounts;
    type Feature = f32;
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

    fn gen_split_feature(&self) -> usize {
        // The data set has four feature columns
        thread_rng().gen_range(0, 4)
    }

    fn train_leaf_predictor(&self) -> ClassCounts {
        // count the number of samples in each class. This is possible
        // because there exists an `impl iter::Sum for ClassCounts`.
        self.iter().map(|sample| sample.y()).sum()
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
        let counts: ClassCounts = self.iter().map(|sample| sample.y()).sum();
        let p1 = counts.probability(Iris(uci_iris::Iris::Setosa));
        let p2 = counts.probability(Iris(uci_iris::Iris::Versicolor));
        let p3 = counts.probability(Iris(uci_iris::Iris::Virginica));
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

/// Here we let the type system know that `Iris` is a categorical type.
impl Categorical for Iris {
    fn as_usize(&self) -> usize {
        self.0 as usize
    }

    fn from_usize(id: usize) -> Self {
        match id {
            0 => Iris(uci_iris::Iris::Setosa),
            1 => Iris(uci_iris::Iris::Versicolor),
            2 => Iris(uci_iris::Iris::Virginica),
            _ => panic!("Invalid class")
        }
    }

    fn n_categories(&self) -> Option<usize> {
        Some(3)
    }
}


fn main() {
    // load data set
    let loader = uci_iris::DataSet::new().create().unwrap();
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
        100,
        DeterministicTreeBuilder::new(
            10,
            None,
            BestRandomSplit::new(4)
        )
    ).fit(&mut training as &mut [_]);

    let mut confusion = [[0; 3]; 3];

    println!("Predicting...");
    for sample in &testing {
        let probs = forest.predict(sample);
        let pred: Iris = probs.most_frequent();

        confusion[sample.y().as_usize()][pred.as_usize()] += 1;
    }

    println!("Confusion matrix:");
    println!("{:>20} : {:>2} {:>2} {:>2}", format!("{:?}", Iris::from_usize(0)), confusion[0][0], confusion[0][1], confusion[0][2]);
    println!("{:>20} : {:>2} {:>2} {:>2}", format!("{:?}", Iris::from_usize(1)), confusion[1][0], confusion[1][1], confusion[1][2]);
    println!("{:>20} : {:>2} {:>2} {:>2}", format!("{:?}", Iris::from_usize(2)), confusion[2][0], confusion[2][1], confusion[2][2]);
}
