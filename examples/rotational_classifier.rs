extern crate forester;
extern crate gnuplot;
extern crate image;
extern crate rand;

use std::{
    f64,
    f64::consts::PI,
    fs::File,
};

use rand::{thread_rng, Rng, ThreadRng};

use forester::{
    traits::{LearnerMut, Predictor},
    criteria::GiniCriterion,
    datasets::TupleSample,
    d_tree::DeterministicTree,
    d_tree::DeterministicTreeBuilder,
    ensemble::EnsembleBuilder,
    features::Mix2,
    predictors::CategoricalProbabilities,
    predictors::ClassPredictor,
    splitters::BestRandomSplit,
    splitters::ThresholdSplitter,
};

//use forester::api::extra_trees_classifier::{ExtraTreesClassifier, Sample};

pub type Sample = TupleSample<Mix2, [f64; 2], u8>;
pub type SplitFitter = BestRandomSplit<ThresholdSplitter<[Sample]>, GiniCriterion<Sample>, ThreadRng>;
pub type Tree = DeterministicTree<ThresholdSplitter<[Sample]>, ClassPredictor<Sample>>;
pub type TreeBuilder = DeterministicTreeBuilder<SplitFitter, ClassPredictor<Sample>>;
pub type Builder = EnsembleBuilder<CategoricalProbabilities, [Sample], TreeBuilder, Tree>;

const N_CLASSES: u8 = 3;
const N_SAMPLES: usize = 1000;

const N_ROWS: u32 = 300;
const N_COLS: u32 = 300;

/// function used to generate training data
fn spiral(r: f64, c: u8) -> f64 {
    let phi = r + PI * 2.0 * c as f64 / N_CLASSES as f64;
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
    let y0: Vec<u8> = (0..N_CLASSES).cycle().take(N_SAMPLES).collect();
    let r = randspace(0.0, 6.0, N_SAMPLES);
    let phi: Vec<_> = r.iter().map(|&r| r).zip(y0.iter())
        .map(|(r, &c)| spiral(r, c) + thread_rng().gen::<f64>() * 2.0 * PI / N_CLASSES as f64)
        .collect();

    let x0: Vec<_> = r.into_iter()
        .zip(phi.into_iter())
        //.zip((0..N_CLASSES).cycle())
        .map(|(r, phi)| [phi.sin() * r, phi.cos() * r]).collect();


    // convert data to data set for fitting
    let mut data: Vec<_> = x0.iter().zip(y0.iter()).map(|(&x, &y)| Sample::new(x, y)).collect();

    // configure and fit random forest
    println!("Fitting...");
    /*let forest = ExtraTreesClassifier::new()
        .n_estimators(100)
        .n_splits(100)
        .min_samples_split(2)
        .fit(&mut data);*/
    let forest = Builder::new(
        100,
        TreeBuilder::new(
            SplitFitter::new(100, thread_rng()),
            2
        )
    ).fit(&mut data);

    // generate test data
    let x_grid = linspace(-4.0, 4.0, N_ROWS as usize);
    let y_grid = linspace(-4.0, 4.0, N_COLS as usize);

    // predict
    println!("Predicting...");
    let mut z = Vec::with_capacity(3 * (N_ROWS * N_COLS) as usize);
    for &y in y_grid.iter() {
        for &x in x_grid.iter() {
            let p = forest.predict(&[x, y]);
            let mut r = p.prob(0);
            let mut g = p.prob(1);
            let mut b = p.prob(2);
            z.push(r);
            z.push(g);
            z.push(b);
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
