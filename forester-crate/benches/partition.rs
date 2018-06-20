#[macro_use]
extern crate criterion;
extern crate rand;
extern crate forester;

use criterion::Criterion;
use criterion::Bencher;

use forester::array_ops::Partition;

fn safe_partition<T, F: FnMut(&T) -> bool>(x: &mut [T], mut predicate: F) -> usize {
    let mut target_index = 0;
    for i in 0..x.len() {
        if predicate(&x[i]) {
            x.swap(target_index, i);
            target_index += 1;
        }
    }
    target_index
}


fn bench_partition(c: &mut Criterion) {
    fn function1(b: &mut Bencher) {
        let x: Vec<_> = (1..1000).collect();
        b.iter(|| {
            x.clone().partition(|&i| i % 2 == 0);
        })
    }
    c.bench_function("Partition (unsafe)", function1);


    fn function2(b: &mut Bencher) {
        let x: Vec<_> = (1..1000).collect();
        b.iter(|| {
            safe_partition(&mut x.clone(), |&i| i % 2 == 0);
        })
    }
    c.bench_function("Partition (safe)", function2);
}

criterion_group!(benches, bench_partition);
criterion_main!(benches);
