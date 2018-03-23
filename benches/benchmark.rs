#[macro_use]
extern crate criterion;
extern crate rand;
extern crate forest;

use std::f64;

use criterion::Criterion;

use rand::{thread_rng, Rng};

use forest::api::ExtraTreesRegressor::*;
use forest::LearnerMut;

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n-1) + fibonacci(n-2),
    }
}

fn gen_data_array(n: usize) -> Vec<Sample<[f64; 30], f64>> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = [0.0; 30];
        for i in 0..x.len() {
            x[i] = thread_rng().gen();
        }
        x[0] = i as f64 / f64::consts::PI;
        let y = x[0].sin();
        data.push(Sample::new(x, y))
    }
    data
}

fn gen_data_vec(n: usize) -> Vec<Sample<Vec<f64>, f64>> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = vec![0.0; 30];
        for i in 0..x.len() {
            x[i] = thread_rng().gen();
        }
        x[0] = i as f64 / f64::consts::PI;
        let y = x[0].sin();
        data.push(Sample::new(x, y))
    }
    data
}


fn array_data(c: &mut Criterion) {
    let mut data = gen_data_array(100);
    c.bench_function("array_data", move |b| b.iter(|| {
        let model = Builder::default().fit(&mut data);
    }));
}

fn vec_data(c: &mut Criterion) {
    let mut data = gen_data_vec(100);
    c.bench_function("vec_data", move |b| b.iter(|| {
        let model = Builder::default().fit(&mut data);
    }));
}

criterion_group!(benches, array_data, vec_data);
criterion_main!(benches);
