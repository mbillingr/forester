use std::cmp;
use std::iter::Sum;
use std::marker::PhantomData;

use num_traits::Bounded;

use rand::thread_rng;
use rand::distributions::range::SampleRange;

use vec2d::Vec2D;


pub mod extra_trees_regressor {
    use super::*;
    use std::f64;
    use rand::Rng;
    use criterion::VarianceCriterion;
    use data::{SampleDescription, TrainingData};
    use dforest::{DeterministicForest, DeterministicForestBuilder};
    use dtree::DeterministicTreeBuilder;
    use iter_mean::IterMean;
    use split::BestRandomSplit;
    use split_between::SplitBetween;

    #[derive(Debug, Clone)]
    pub struct Sample<'a, X: 'a, Y> {
        x: &'a[X],
        y: Y,
    }

    impl<'a, X: 'a, Y> Sample<'a, X, Y> {
        pub fn new(x: &'a[X], y: Y) -> Self {
            Sample { x, y }
        }
    }

    impl<'a, X, Y> SampleDescription for Sample<'a, X, Y>
        where X: Clone + PartialOrd + SampleRange + SplitBetween,
              Y: Clone
    {
        type ThetaSplit = usize;
        type ThetaLeaf = f64;
        type Feature = X;
        type Target = Y;
        type Prediction = f64;

        fn target(&self) -> Self::Target {
            self.y.clone()
        }

        fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
            self.x[*theta].clone()
        }

        fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction {
            *w
        }
    }

    impl<'a, X> TrainingData<Sample<'a, X, f64>> for [Sample<'a, X, f64>]
        where X: Clone + PartialOrd + SampleRange + Bounded + SplitBetween
    {
        type Criterion = VarianceCriterion;
        fn n_samples(&self) -> usize {
            self.len()
        }

        fn gen_split_feature(&self) -> usize {
            let n = self[0].x.len();
            thread_rng().gen_range(0, n)
        }

        fn train_leaf_predictor(&self) -> f64 {
            f64::mean(self.iter().map(|sample| &sample.y))
        }

        fn feature_bounds(&self, theta: &usize) -> (X, X) {
            self.iter()
                .map(|sample| sample.sample_as_split_feature(theta))
                .fold((X::max_value(), X::min_value()),
                      |(min, max), x| {
                          (if x < min {x.clone()} else {min},
                           if x > max {x} else {max})
                      })
        }
    }

    pub struct ExtraTreesRegressor {
        n_estimators: usize,
        n_splits: usize,
        min_samples_split: usize,
        max_depth: Option<usize>,
        bootstrap: Option<usize>,
    }

    impl ExtraTreesRegressor {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_n_estimators(mut self, n: usize) -> Self {
            self.n_estimators = n;
            self
        }

        pub fn with_n_splits(mut self, n: usize) -> Self {
            self.n_splits = n;
            self
        }

        pub fn with_min_samples_split(mut self, n: usize) -> Self {
            self.min_samples_split = n;
            self
        }

        pub fn with_max_depth(mut self, d: usize) -> Self {
            self.max_depth = Some(d);
            self
        }

        pub fn with_bootstrap(mut self, d: usize) -> Self {
            self.max_depth = Some(d);
            self
        }

        pub fn fit<'a, 'b, T>(&'a self, x: &'b Vec2D<T>, y: &'b Vec<f64>) -> DeterministicForest<Sample<'b, T, f64>>
            where T: Clone + cmp::PartialOrd + SampleRange + Bounded + SplitBetween,
        {
            let mut data: Vec<Sample<T, f64>> = x.iter()
                .zip(y.iter())
                .map(|(xi, yi)| Sample{x: xi, y: *yi})
                .collect();

            DeterministicForestBuilder::new(
                self.n_estimators,
                DeterministicTreeBuilder {
                    _p: PhantomData,
                    min_samples_split: self.min_samples_split,
                    min_samples_leaf: 1,
                    split_finder: BestRandomSplit::new(self.n_splits),
                    max_depth: self.max_depth,
                    bootstrap: self.bootstrap,
                }
            ).fit(&mut data[..])
        }
    }

    impl Default for ExtraTreesRegressor {
        fn default() -> Self {
            Self {
                n_estimators: 10,
                n_splits: 1,
                min_samples_split: 2,
                max_depth: None,
                bootstrap: None,
            }
        }
    }
}


pub mod extra_trees_classifier {
    use super::*;
    use std::f64;
    use rand::Rng;
    use categorical::{Categorical, CatCount};
    use criterion::GiniCriterion;
    use data::{SampleDescription, TrainingData};
    use dforest::{DeterministicForest, DeterministicForestBuilder};
    use dtree::DeterministicTreeBuilder;
    use iter_mean::IterMean;
    use split::BestRandomSplit;
    use split_between::SplitBetween;

    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub struct Classes(pub u8);

    impl Categorical for Classes {
        fn as_usize(&self) -> usize {
            self.0 as usize
        }

        fn from_usize(id: usize) -> Self {
            assert!(id < 256);
            Classes(id as u8)
        }

        fn n_categories(&self) -> Option<usize> {
            // We don't know the total number of classes
            None
        }
    }

    #[derive(Debug, Clone)]
    pub struct ClassCounts {
        counts: Vec<usize>,
        total: usize,
    }

    impl ClassCounts {
        fn new() -> Self {
            ClassCounts {
                counts: Vec::new(),
                total: 0,
            }
        }

        pub fn probs<F: FnMut(f64)>(&self, mut f: F) {
            let n = self.total as f64;
            for c in self.counts.iter() {
                f(*c as f64 / n);
            }
        }
    }

    impl CatCount<Classes> for ClassCounts {
        fn add(&mut self, c: Classes) {
            self.add_n(c, 1)
        }

        fn add_n(&mut self, c: Classes, n: usize) {
            let i = c.as_usize();
            if i >= self.counts.len() {
                self.counts.resize(i + 1, 0);
            }
            self.counts[i] += n;
            self.total += n;
        }

        fn probability(&self, c: Classes) -> f64 {
            let i = c.as_usize();
            if i < self.counts.len() {
                self.counts[i] as f64 / self.total as f64
            } else {
                0.0
            }
        }

        fn most_frequent(&self) -> Classes {
            let mut n = 0;
            let mut c = 0;
            for i in 0..self.counts.len() {
                // TODO: handle ties?
                if self.counts[i] > n {
                    n = self.counts[i];
                    c = i;
                }
            }
            Classes::from_usize(c)
        }
    }

    impl<'a> Sum<&'a Classes> for ClassCounts {
        fn sum<I: Iterator<Item=&'a Classes>>(iter: I) -> Self {
            let mut counts = ClassCounts::new();
            for c in iter {
                counts.add(*c);
            }
            counts
        }
    }

    impl IterMean<ClassCounts> for ClassCounts {
        fn mean<I: ExactSizeIterator<Item=ClassCounts>>(iter: I) -> Self {
            let mut total_counts = ClassCounts::new();
            for c in iter {
                for (i, n) in c.counts.iter().enumerate() {
                    total_counts.add_n(Classes(i as u8), *n);
                }
            }
            total_counts
        }
    }

    #[derive(Debug, Clone)]
    pub struct Sample<'a, X: 'a, Y>
        where X: Clone + PartialOrd + SampleRange,
    {
        x: &'a[X],
        y: Y,
    }

    impl<'a, X: 'a, Y> Sample<'a, X, Y>
        where X: Clone + PartialOrd + SampleRange,
    {
        pub fn new(x: &'a[X], y: Y) -> Self {
            Sample { x, y }
        }
    }

    impl<'a, X, Y> SampleDescription for Sample<'a, X, Y>
        where X: Clone + PartialOrd + SampleRange + SplitBetween,
              Y: Clone
    {
        type ThetaSplit = usize;
        type ThetaLeaf = ClassCounts;
        type Feature = X;
        type Target = Y;
        type Prediction = ClassCounts;

        fn target(&self) -> Self::Target {
            self.y.clone()
        }

        fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
            self.x[*theta].clone()
        }

        fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction {
            w.clone()
        }
    }

    impl<'a, X> TrainingData<Sample<'a, X, Classes>> for [Sample<'a, X, Classes>]
        where X: Clone + PartialOrd + SampleRange + Bounded + SplitBetween
    {
        type Criterion = GiniCriterion;

        fn n_samples(&self) -> usize {
            self.len()
        }

        fn gen_split_feature(&self) -> usize {
            let n = self[0].x.len();
            thread_rng().gen_range(0, n)
        }

        fn train_leaf_predictor(&self) -> ClassCounts {
            self.iter().map(|sample| &sample.y).sum()
        }

        fn feature_bounds(&self, theta: &usize) -> (X, X) {
            self.iter()
                .map(|sample| sample.sample_as_split_feature(theta))
                .fold((X::max_value(), X::min_value()),
                      |(min, max), x| {
                          (if x < min {x.clone()} else {min},
                           if x > max {x} else {max})
                      })
        }
    }

    pub struct ExtraTreesClassifier {
        n_estimators: usize,
        n_splits: usize,
        min_samples_split: usize,
        max_depth: Option<usize>,
        bootstrap: Option<usize>,
    }

    impl ExtraTreesClassifier {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_n_estimators(mut self, n: usize) -> Self {
            self.n_estimators = n;
            self
        }

        pub fn with_n_splits(mut self, n: usize) -> Self {
            self.n_splits = n;
            self
        }

        pub fn with_min_samples_split(mut self, n: usize) -> Self {
            self.min_samples_split = n;
            self
        }

        pub fn with_max_depth(mut self, d: usize) -> Self {
            self.max_depth = Some(d);
            self
        }

        pub fn with_bootstrap(mut self, d: usize) -> Self {
            self.max_depth = Some(d);
            self
        }

        pub fn fit<'a, 'b, T>(&'a self, x: &'b Vec2D<T>, y: &'b Vec<u8>) -> DeterministicForest<Sample<'b, T, Classes>>
            where T: Clone + cmp::PartialOrd + SampleRange + Bounded + SplitBetween,
        {
            let mut data: Vec<Sample<T, Classes>> = x.iter()
                .zip(y.iter())
                .map(|(xi, yi)| Sample{x: xi, y: Classes(*yi)})
                .collect();

            DeterministicForestBuilder::new(
                self.n_estimators,
                DeterministicTreeBuilder {
                    _p: PhantomData,
                    min_samples_split: self.min_samples_split,
                    min_samples_leaf: 1,
                    split_finder: BestRandomSplit::new(self.n_splits),
                    max_depth: self.max_depth,
                    bootstrap: self.bootstrap,
                }
            ).fit(&mut data[..])
        }
    }

    impl Default for ExtraTreesClassifier {
        fn default() -> Self {
            Self {
                n_estimators: 10,
                n_splits: 1,
                min_samples_split: 2,
                max_depth: None,
                bootstrap: None,
            }
        }
    }
}


#[cfg(test)]
mod tests {

    #[test]
    fn extra_trees_regressor() {
        use super::extra_trees_regressor::{ExtraTreesRegressor, Sample};
        use vec2d::Vec2D;

        let x = Vec2D::from_slice(&[1, 2, 3,    7, 8, 9], 1);
        let y = vec![5.0, 5.0, 5.0,    2.0, 2.0, 2.0];

        let model = ExtraTreesRegressor::new()
            .with_n_estimators(2)
            .with_n_splits(1)
            .with_min_samples_split(2)
            .with_max_depth(3)
            .fit(&x, &y);

        assert_eq!(model.predict(&Sample::new(&[-1000], ())), 5.0);
        assert_eq!(model.predict(&Sample::new(&[1000], ())), 2.0);

        let p = model.predict(&Sample::new(&[5], ()));
        assert!(p >= 2.0);
        assert!(p <= 5.0);
    }

    #[test]
    fn extra_trees_classifier() {
        use super::extra_trees_classifier::Classes;
        use super::extra_trees_classifier::ExtraTreesClassifier;
        use super::extra_trees_classifier::Sample;
        use categorical::{Categorical, CatCount};
        use vec2d::Vec2D;

        assert_eq!(Classes(42).n_categories(), None);

        let x = Vec2D::from_slice(&[1, 2, 3, 7, 8, 9], 1);
        let y = vec![1, 1, 1, 2, 2, 2];

        let model = ExtraTreesClassifier::new()
            .with_n_estimators(100)
            .with_n_splits(1)
            .with_min_samples_split(2)
            .with_max_depth(5)
            .fit(&x, &y);

        assert_eq!(model.predict(&Sample::new(&[-1000], ())).probability(Classes(1)), 1.0);
        assert_eq!(model.predict(&Sample::new(&[1000], ())).probability(Classes(2)), 1.0);

        let p = model.predict(&Sample::new(&[5], ()));
        assert_eq!(p.probability(Classes(0)), 0.0);
        assert!(p.probability(Classes(1)) > 0.0);
        assert!(p.probability(Classes(2)) > 0.0);
        assert_eq!(p.probability(Classes(3)), 0.0);

        assert_eq!(model.predict(&Sample::new(&[2], ())).most_frequent(), Classes(1));
        assert_eq!(model.predict(&Sample::new(&[8], ())).most_frequent(), Classes(2));
    }
}
