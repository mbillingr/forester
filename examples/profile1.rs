extern crate forester;
extern crate rand;

pub fn main() {
    #[cfg(feature = "cpuprofiler")] {
        extern crate cpuprofiler;
        use std::f64::consts::PI;

        use rand::{thread_rng, Rng};

        use forester::traits::Predictor;
        use forester::api::extra_trees_regressor::{ExtraTreesRegressor, Sample};

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

        let x0 = randspace(0.0, 1.0, 1000);
        let y0: Vec<_> = x0.iter().map(func).collect();

        let mut data: Vec<_> = x0.iter().zip(y0.iter()).map(|(&x, &y)| Sample::new([x], y)).collect();

        let fb = ExtraTreesRegressor::new()
            .n_estimators(100)
            .n_splits(100)
            .min_samples_split(2);

        cpuprofiler::PROFILER.lock().unwrap().start("fit.profile").unwrap();
        let forest = fb.fit(&mut data);
        cpuprofiler::PROFILER.lock().unwrap().stop().unwrap();

        let x = linspace(-0.2, 1.2, 10000);

        cpuprofiler::PROFILER.lock().unwrap().start("predict.profile").unwrap();
        let y: Vec<_> = x.iter().map(|&x| forest.predict(&[x])).collect();
        cpuprofiler::PROFILER.lock().unwrap().stop().unwrap();

        println!("Profiling done. Convert the profile with something like");
        println!("  > pprof --callgrind target/debug/examples/profile1 fit.profile > fit.prof");
        println!("Or view it with\n  > pprof --gv target/debug/examples/profile1 fit.profile");
    }
}
