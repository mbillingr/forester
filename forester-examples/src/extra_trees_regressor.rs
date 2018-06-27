extern crate examples_common;
extern crate forester;
extern crate gnuplot;
extern crate rand;

use std::f64::consts::PI;

use gnuplot::*;

use rand::{thread_rng, Rng};

use forester::api::extra_trees_regressor::{ExtraTreesRegressor, Sample};
use forester::vec2d::Vec2D;

use examples_common::numeric::{linspace, randspace};

/// function used to generate training data
fn func(x: &f64) -> f64 {
    (x * 2.0 * PI).sin() + thread_rng().gen::<f64>() * 0.2
}

fn main() {
    // generate data points
    let x0 = randspace(0.0, 1.0, 100);
    let y0: Vec<_> = x0.iter().map(func).collect();

    let x_train = Vec2D::from_slice(&x0, 1);
    let y_train = y0.clone();

    // configure and fit random forest
    let forest = ExtraTreesRegressor::new()
        .with_n_estimators(100)
        .with_n_splits(1)
        .with_min_samples_split(10)
        .fit(&x_train, &y_train);

    // generate test data
    let x = linspace(-0.2, 1.2, 1000);

    // predict y values
    let y: Vec<_> = x.iter().map(|&x| forest.predict(&Sample::new(&[x], ()))).collect();

    // plot results
    let mut fig = Figure::new();
    fig.axes2d()
        .set_title("Extra Trees Regressor", &[])
        .points(&x0, &y0, &[Color("red"), PointSymbol('O')])
        .lines(&x, &y, &[Color("blue")]);

    fig.show();
}
