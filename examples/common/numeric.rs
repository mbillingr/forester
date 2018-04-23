
use rand::{thread_rng, Rng};

/// generate a Vec<f64> of linearly spaced values
pub fn linspace(l: f64, h: f64, n: usize) -> Vec<f64> {
    let di = (h - l) / (n - 1) as f64;
    (0..n).map(|i| l + di * i as f64).collect()
}

/// generate a Vec<f64> of uniformly distributed random values
pub fn randspace(l: f64, h: f64, n: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    (0..n).map(|_| rng.gen_range(l, h)).collect()
}
