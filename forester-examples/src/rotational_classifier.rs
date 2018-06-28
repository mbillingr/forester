extern crate examples_common;
extern crate forester;
extern crate image;
extern crate rand;

use std::f64::consts::PI;
use std::fs::File;

use rand::{thread_rng, Rng};

use forester::categorical::CatCount;
use forester::criterion::GiniCriterion;
use forester::data::{SampleDescription, TrainingData};
use forester::dforest::DeterministicForestBuilder;
use forester::dtree::DeterministicTreeBuilder;
use forester::split::BestRandomSplit;

use examples_common::rgb_classes::{ClassCounts, Classes};

#[derive(Clone)]
struct Sample<Y> {
    x: [f64; 2],
    y: Y,
}

impl<Y: Copy> SampleDescription for Sample<Y> {
    type ThetaSplit = (f64, f64);
    type ThetaLeaf = ClassCounts;
    type Feature = f64;
    type Target = Y;
    type Prediction = ClassCounts;

    fn target(&self) -> Self::Target {
        self.y
    }

    fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
        self.x[0] * theta.0 + self.x[1] * theta.1
    }

    fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction {
        w.clone()
    }
}

impl TrainingData<Sample<Classes>> for [Sample<Classes>]
{
    type Criterion = GiniCriterion; // TODO: specialized Gini implementation for three classes

    fn n_samples(&self) -> usize {
        self.len()
    }

    fn gen_split_feature(&self) -> (f64, f64) {
        let a: f64 = thread_rng().gen::<f64>() * PI;
        (a.sin(), a.cos())
    }

    fn train_leaf_predictor(&self) -> ClassCounts {
        self.iter().map(|sample| sample.y).sum()
    }

    fn feature_bounds(&self, theta: &(f64, f64)) -> (f64, f64) {
        self.iter()
            .map(|sample| sample.sample_as_split_feature(theta))
            .fold((std::f64::INFINITY, std::f64::NEG_INFINITY),
                         |(min, max), x| {
                             (if x < min {x} else {min},
                              if x > max {x} else {max})
        })
    }
}

const N_SAMPLES: usize = 1000;

const N_ROWS: u32 = 300;
const N_COLS: u32 = 300;

/// function used to generate training data
fn spiral(r: f64, c: u8) -> f64 {
    let phi = r + PI * 2.0 * c as f64 / 3.0;
    phi
}

/// generate a Vec<f64> of linearly spaced values
fn linspace(l: f64, h: f64, n: usize) -> Vec<f64> {
    let di = (h - l) / (n - 1) as f64;
    (0..n).map(|i| l + di * i as f64).collect()
}

/// generate a Vec<f64> of uniformly distributed random values
fn randspace(l: f64, h: f64, n: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    (0..n).map(|_| rng.gen_range(l, h)).collect()
}

fn main() {
    // generate data points
    let y0: Vec<Classes> = vec![Classes::Red, Classes::Green, Classes::Blue].into_iter().cycle().take(N_SAMPLES).collect();
    let r = randspace(0.0, 6.0, N_SAMPLES);
    let phi: Vec<_> = r.iter().map(|&r| r).zip(y0.iter())
        .map(|(r, &c)| spiral(r, c as u8) + thread_rng().gen::<f64>() * PI * 2.0 / 3.0)
        .collect();

    let x0: Vec<_> = r.into_iter()
        .zip(phi.into_iter())
        .map(|(r, phi)| [phi.sin() * r, phi.cos() * r]).collect();

    // convert data to data set for fitting
    let mut data: Vec<_> = x0.iter().zip(y0.iter()).map(|(&x, &y)| Sample{x, y}).collect();

    // configure and fit random forest
    println!("Fitting...");
    let forest = DeterministicForestBuilder::new(
        100,
        DeterministicTreeBuilder::new(
            2,
            BestRandomSplit::new(100)
        )
    ).fit(&mut data as &mut [_]);

    // generate test data
    let x_grid = linspace(-4.0, 4.0, N_ROWS as usize);
    let y_grid = linspace(-4.0, 4.0, N_COLS as usize);

    // predict
    println!("Predicting...");
    let mut z = Vec::with_capacity(3 * (N_ROWS * N_COLS) as usize);
    for &y in y_grid.iter() {
        for &x in x_grid.iter() {
            let sx = [x, y];
            let c = forest.predict(&Sample{x: sx, y: ()});
            z.push(c.probability(Classes::Red));
            z.push(c.probability(Classes::Green));
            z.push(c.probability(Classes::Blue));
        }
    }

    // plot original samples
    for xy in x0 {
        let (x, y) = (xy[0], xy[1]);
        let x = (N_COLS as f64 * (x + 4.0) / 8.0) as usize;
        let y = (N_ROWS as f64 * (y + 4.0) / 8.0) as usize;

        if x <= 0 || y <= 0 || x >= N_COLS as usize - 1 || y >= N_ROWS as usize - 1 {
            continue
        }

        z[(x + y * N_COLS as usize) * 3 + 0] *= 0.5;
        z[(x + y * N_COLS as usize) * 3 + 1] *= 0.5;
        z[(x + y * N_COLS as usize) * 3 + 2] *= 0.5;

        z[(x + 1 + y * N_COLS as usize) * 3 + 0] *= 0.5;
        z[(x + 1 + y * N_COLS as usize) * 3 + 1] *= 0.5;
        z[(x + 1 + y * N_COLS as usize) * 3 + 2] *= 0.5;

        z[(x - 1 + y * N_COLS as usize) * 3 + 0] *= 0.5;
        z[(x - 1 + y * N_COLS as usize) * 3 + 1] *= 0.5;
        z[(x - 1 + y * N_COLS as usize) * 3 + 2] *= 0.5;

        z[(x + (y + 1) * N_COLS as usize) * 3 + 0] *= 0.5;
        z[(x + (y + 1) * N_COLS as usize) * 3 + 1] *= 0.5;
        z[(x + (y + 1) * N_COLS as usize) * 3 + 2] *= 0.5;

        z[(x + (y - 1) * N_COLS as usize) * 3 + 0] *= 0.5;
        z[(x + (y - 1) * N_COLS as usize) * 3 + 1] *= 0.5;
        z[(x + (y - 1) * N_COLS as usize) * 3 + 2] *= 0.5;
    }

    // store result
    let z: Vec<u8> = z.into_iter().map(|i| (i * 255.0) as u8).collect();
    let encoder = image::png::PNGEncoder::new(File::create("rotational_classifier.png").unwrap());
    encoder.encode(&z, N_COLS, N_ROWS, image::ColorType::RGB(8)).unwrap();
}
