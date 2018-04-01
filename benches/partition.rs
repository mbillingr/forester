#[macro_use]
extern crate criterion;
extern crate rand;
extern crate forester;

use std::f64;

use criterion::Criterion;
use criterion::Bencher;

use forester::array_ops::Partition;


fn bench_partition(c: &mut Criterion) {
    fn function(b: &mut Bencher) {
        let mut x: Vec<_> = (1..1000).collect();
        b.iter(|| {
            x.clone().partition(|&i| i % 2 == 0);
        })
    }

    c.bench_function("Partition", function);
}

criterion_group!(benches, compare_fibonaccis);
criterion_main!(benches);
