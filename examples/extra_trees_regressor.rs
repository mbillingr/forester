extern crate forest;
extern crate gnuplot;
extern crate rand;

use std::f64::consts::PI;

use gnuplot::*;

use rand::{thread_rng, Rng};

use forest::traits::*;
use forest::api::extra_trees_regressor as etr;

fn func(x: &f64) -> f64 {
    (x * 2.0 * PI).sin() + thread_rng().gen::<f64>() * 0.2
}

fn linspace(l: f64, h: f64, n: usize) -> Vec<f64> {
    let di = (h - l) / (n - 1) as f64;
    (0..n).map(|i| l + di * i as f64).collect()
}

fn randspace(l: f64, h: f64, n: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    (0..n).map(|_| rng.gen_range(l, h)).collect()
}

fn main() {

    let x0 = randspace(0.0, 1.0, 100);
    let y0: Vec<_> = x0.iter().map(func).collect();

    let mut data: Vec<etr::Sample<_, _>> = x0.iter().zip(y0.iter()).map(|(&x, &y)| etr::Sample::new([x], y)).collect();
    let forest = etr::Builder::new(
        100,
        etr::TreeBuilder::new(
            etr::SplitFitter::new(1, thread_rng()),
            10,
        )
    ).fit(&mut data);

    let x = linspace(-0.2, 1.2, 1000);
    let y: Vec<_> = x.iter().map(|&x| forest.predict(&[x])).collect();

    let mut fig = Figure::new();
    fig.axes2d()
        .set_title("Extra Trees Regressor", &[])
        .points(&x0, &y0, &[Color("red"), PointSymbol('O')])
        .lines(&x, &y, &[Color("blue")]);

    fig.show();
}
