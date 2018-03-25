#[macro_use]
extern crate criterion;
extern crate rand;
extern crate forest;

use std::f64;

use criterion::Criterion;

use rand::{thread_rng, Rng};

use forest::api::ExtraTreesRegressor::*;
use forest::LearnerMut;

const N: usize = 100;
const M: usize = 30;

static mut g_data: Option<Vec<f64>> = None;

fn get_data() -> &'static Vec<f64> {
    unsafe {
        match g_data {
            Some(ref d) => return d,
            None => { }
        }
        let mut x = Vec::with_capacity(N*M);
        for i in 0..N*M {
            x.push(thread_rng().gen());
        }
        g_data = Some(x);
        g_data.as_ref().unwrap()
    }
}

fn gen_data_array() -> Vec<Sample<[f64; M], f64>> {
    let mut data = Vec::with_capacity(N);
    for gx in get_data().chunks(M) {
        let mut x = [0.0; M];
        for i in 0..M {
            x[i] = gx[i];
        }
        let y = x[0].sin();
        data.push(Sample::new(x, y))
    }
    data
}

fn gen_data_vec() -> Vec<Sample<Vec<f64>, f64>> {
    let mut data = Vec::with_capacity(N);
    for gx in get_data().chunks(M) {
        let x: Vec<f64> = gx.iter().map(|i| *i).collect();
        let y = x[0].sin();
        data.push(Sample::new(x, y))
    }
    data
}

fn gen_data_ref() -> Vec<Sample<&'static [f64], f64>> {
    let mut data = Vec::with_capacity(N);
    for x in get_data().chunks(M) {
        let y = x[0].sin();
        data.push(Sample::new(x, y))
    }
    data
}


fn array_data(c: &mut Criterion) {
    let mut data = gen_data_array();
    c.bench_function("array_data", move |b| b.iter(|| {
        let model = TreeBuilder::default().fit(&mut data);
    }));
}

fn vec_data(c: &mut Criterion) {
    let mut data = gen_data_vec();
    c.bench_function("vec_data", move |b| b.iter(|| {
        let model = TreeBuilder::default().fit(&mut data);
    }));
}

fn ref_data(c: &mut Criterion) {
    let mut data = gen_data_ref();
    c.bench_function("ref_data", move |b| b.iter(|| {
        let model = TreeBuilder::default().fit(&mut data);
    }));
}

criterion_group!(benches, array_data, ref_data, vec_data);
criterion_main!(benches);
