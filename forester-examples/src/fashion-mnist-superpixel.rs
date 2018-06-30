/// This example shows how to construct a random forest classifier that generates features from
/// a large feature space on the fly.
///
/// We will be using the MNIST dataset with super-pixels as features. For the purpose of this
/// example, let's define super-pixel as the rectangular region of a greyscale image that takes the
/// maximum pixel value from inside the region. We can use those as features instead of one feature
/// per pixel, which drastically increases the number of features. In the MNIST set we have 28x28
/// pixels. There are 28x28 possible top-left corner choices for a super pixel, and (28-t)x(28-l)
/// choices of the top-right corners *for each* top left corner. The total number of possible
/// super-pixels is 164836.
///
/// We certainly do not want to compute and store them all - the data set would have more than 10GB!
/// Instead, the random forest can generate the features on the fly when needed. This example shows
/// hot to accomplish that with the `forester` crate.


extern crate examples_common;
extern crate forester;
extern crate openml;
extern crate rand;
#[macro_use]
extern crate serde_derive;

use rand::{thread_rng, Rng};
use openml::MeasureAccumulator;

use forester::categorical::CatCount;
use forester::criterion::GiniCriterion;
use forester::data::{SampleDescription, TrainingData};
use forester::dforest::DeterministicForestBuilder;
use forester::dtree::DeterministicTreeBuilder;
use forester::split::{BestSplitRandomFeature, Split};

use examples_common::dig_classes::ClassCounts;

/// This struct holds the pixel data of one image
#[derive(Debug, Deserialize)]
struct Pixels(Vec<u8>);

/// A sample is one image `x` with a class label `y` attached. We will use this struct for training
/// on labelled data and for training, where we estimate the label (`y` has a dummy value in this
/// case).
#[derive(Debug, Copy, Clone)]
struct Sample<'a> {
    x: &'a Pixels,
    y: u8,
}

// for convenient construction of samples
impl<'a> From<(&'a Pixels, &'a u8)> for Sample<'a> {
    fn from(src: (&'a Pixels, &'a u8)) -> Self {
        Sample {
            x: src.0,
            y: *src.1,
        }
    }
}

/// Value(s) that identifies a feature.
///
/// In the case of super pixels, this is a rectangular region of the source image. Instead of
/// storing the top/left/bottom/right coordinates, a representation is chosen that is better suited
/// for getting the data out of a flattened `Pixels` array.
#[derive(Debug, Copy, Clone)]
struct SuperPixel {
    /// the flattened index of the first (top-left) pixel
    offset: u16,

    /// 1 + the flattened index of the last (bottom-right) pixel
    end: u16,

    /// the row length or width of the region
    rowlen: u8,

    /// number of indices to skip when advancing from one row to the next
    stride: u8,
}

impl SuperPixel {
    /// Construct a super pixel from its borders
    fn new(left: i32, right: i32, top: i32, bot: i32) -> Self {
        SuperPixel {
            offset: (left + top * 28) as u16,
            end: 1 + (right + bot * 28) as u16,
            rowlen: (1 + right - left) as u8,
            stride: 28,
        }
    }
}

/// Implementing the `SampleDescription` trait allows `forester` to make sense of our custom
/// `Sample` type.
impl<'a> SampleDescription for Sample<'a> {
    /// the type that parameterizes a feature for splitting
    type ThetaSplit = SuperPixel;

    /// what we store in each leaf; in this case the number of samples falling in each class
    type ThetaLeaf = ClassCounts;

    /// the type of a feature value
    type Feature = u8;

    /// the type of a target value
    type Target = u8;

    /// the type that we predict; often this is just the same as `ThetaLeaf`.
    type Prediction = ClassCounts;

    /// extract target value of a sample
    fn target(&self) -> Self::Target {
        self.y
    }

    /// extract feature value of a sample given its parametrization. Here we compute the maximum
    /// pixel value in the given super pixel
    fn sample_as_split_feature(&self, theta: &SuperPixel) -> Self::Feature {
        let mut max = 0;

        let mut i = theta.offset as usize;
        while i < theta.end as usize {
            for x in self.x.0[i..i+theta.rowlen as usize].iter() {
                max = max.max(*x);
            }
            i += theta.stride as usize;
        }

        max
    }

    /// to predict the target value of a sample we simply clone the data that is stored in the
    /// leaf node this sample falls into.
    fn sample_predict(&self, c: &Self::ThetaLeaf) -> Self::Prediction {
        c.clone()
    }
}

/// Implementing the `TrainingData` trait allows `forester` to make sense of a data set; a
/// collection of `Sample`s.
impl<'a> TrainingData<Sample<'a>> for [Sample<'a>] {
    /// the performance criterion used for evaluating splits
    type Criterion = GiniCriterion;

    /// number of samples in the data set
    fn n_samples(&self) -> usize {
        self.len()
    }

    /// generate a randomized feature parameter. This is used by splitting strategies that do not
    /// care if a feature is used more than once.
    /// The current implementation simply generates two random points and defines the rectangle they
    /// span as the super-pixel feature. This favors the central regions of the image. Since the
    /// central region is probably more interesting than the borders anyway I'll leave it at that.
    fn gen_split_feature(&self) -> SuperPixel {
        let mut rng = thread_rng();

        let a = rng.gen_range(0, 28);
        let b = rng.gen_range(0, 28);
        let c = rng.gen_range(0, 28);
        let d = rng.gen_range(0, 28);

        let left = i32::min(a, b);
        let right = i32::max(a, b);
        let top = i32::min(c, d);
        let bot = i32::max(c, d);

        SuperPixel::new(left, right, top, bot)
    }

    /// generate an iterator over all possible features parameters. This is used by splitting
    /// strategies that want to try no feature more than once.
    fn all_split_features(&self) -> Option<Box<Iterator<Item=SuperPixel>>> {
        //return Some(Box::new((0..784).map(|i| SuperPixel { offset: i, end: i+1, stride: 28, rowlen: 1})));

        let left = 0..28;
        let top = 0..28;

        let iter = (0..28)
            .flat_map(|t|
                (0..28).flat_map( move |l|
                    (t..28.min(t+5)).flat_map(move |b|
                        (l..28.min(l+5)).map(move |r|
                            SuperPixel::new(l, r, t, b)
                        )
                    )
                )
            );

        Some(Box::new(iter))
    }

    /// how to train the predictors that sit at each leaf of the decision trees.
    /// We just count classes here.
    fn train_leaf_predictor(&self) -> ClassCounts {
        // count the number of samples in each class. This is possible
        // because there exists an `impl iter::Sum for ClassCounts`.
        self.iter().map(|sample| sample.y).sum()
    }

    /// return the minimum and maximum value of a feature. Here we simply use the whole possible
    /// range because `u8` is rather limited. For other data sets it can be beneficial to iterate
    /// over the data to get minimum and maximum.
    fn feature_bounds(&self, _theta: &SuperPixel) -> (u8, u8) {
        (0, 255)
    }
}


fn main() {
    // get "Supervised Classification on Fashion-MNIST" task (https://www.openml.org/t/146825)
    let task = openml::SupervisedClassification::from_openml(146825).unwrap();

    println!("Task: {}", task.name());

    // run the task. This function expects us to provide a closure to which takes an iterator over
    // the training set and another iterator over the testing set, and that returns an iterator
    // over the predicted values. Depending on the task the closure can be called multiple times,
    // once for each cross-validation fold.
    // Furthermore, we specify the evaluation measure we want to use. `Predictive Accuracy` is a
    // simple measure that shows quickly how well the classification works.
    let acc: openml::PredictiveAccuracy<_> = task.run_static(|train, test| {

        // convert the training data in a form that we can use for training the classifier.
        let mut train: Vec<_> = train.map(Sample::from).collect();

        println!("Fitting...");
        // The DeterministicForestBuilder lets us specify options to configure our random forest
        // before fitting.
        let forest = DeterministicForestBuilder::new(
            10,  // number of trees in the forest
            DeterministicTreeBuilder::new(  // the tree builder configures trees :)
                2,  // minimum number of samples required to attempt a split
                BestSplitRandomFeature::new(100) // The `BestSplitRandomFeature` split finder is similar to CART trees. We try 100 features in each split (more would be better but slower).
            ).with_bootstrap(1000) // Fit each tree on 1000 samples randomly taken from the training set with replacement (more would be better but slower).
        ).fit(&mut train as &mut [_]);

        println!("Predicting...");
        // Predict samples by mapping the test set input to predicted class labels.
        let result: Vec<_> = test
            .map(|x| {
                let sample = Sample {
                    x,
                    y: 99  // dummy value for the class label, which is not used by the `predict` function.
                };
                // predict the most likely class label
                let prediction: u8 = forest.predict(&sample).most_frequent();
                prediction
            })
            .collect();

        // return boxed iterator over predictions
        Box::new(result.into_iter())
    });

    // Display the result
    println!("{:#?}", acc);
    println!("{:#?}", acc.result());
}
