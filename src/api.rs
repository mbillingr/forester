use std::cmp;

use num_traits::Bounded;

use rand::{thread_rng, ThreadRng};
use rand::distributions::range::SampleRange;

use criteria::GiniCriterion;
use d_tree::{DeterministicTree, DeterministicTreeBuilder};
use datasets::TupleSample;
use ensemble::{Ensemble, EnsembleBuilder};
use features::ColumnSelect;
use get_item::GetItem;
use predictors::{CategoricalProbabilities, ClassPredictor};
use splitters::{BestRandomSplit, ThresholdSplitter};
use traits::LearnerMut;
use vec2d::Vec2D;


pub mod extra_trees_regressor {
    use std::f64;
    use rand::Rng;
    use super::*;
    use ::{AsTrainingData, BestRandomSplit, DeterministicForest, DeterministicForestBuilder, DeterministicTreeBuilder, SampleDescription, Split, TrainingData};
    use array_ops::Partition;
    use iter_mean::IterMean;

    #[derive(Debug)]
    pub struct Sample<'a, X: 'a, Y> {
        x: &'a[X],
        y: Y,
    }

    impl<'a, X, Y> SampleDescription for Sample<'a, X, Y>
        where X: Clone + PartialOrd + SampleRange
    {
        type ThetaSplit = usize;
        type ThetaLeaf = f64;
        type Feature = X;
        type Prediction = f64;

        fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
            self.x[*theta].clone()
        }

        fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction {
            *w
        }
    }

    // TODO: these impls just scream for macros

    impl<'a, X> SampleDescription for &'a[X]
        where X: Clone + PartialOrd + SampleRange
    {
        type ThetaSplit = usize;
        type ThetaLeaf = f64;
        type Feature = X;
        type Prediction = f64;

        fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
            self[*theta].clone()
        }

        fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction {
            *w
        }
    }

    impl<X> SampleDescription for [X]
        where X: Clone + PartialOrd + SampleRange
    {
        type ThetaSplit = usize;
        type ThetaLeaf = f64;
        type Feature = X;
        type Prediction = f64;

        fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
            self[*theta].clone()
        }

        fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction {
            *w
        }
    }

    impl<X> SampleDescription for [X; 1]
        where X: Clone + PartialOrd + SampleRange
    {
        type ThetaSplit = usize;
        type ThetaLeaf = f64;
        type Feature = X;
        type Prediction = f64;

        fn sample_as_split_feature(&self, theta: &Self::ThetaSplit) -> Self::Feature {
            self[*theta].clone()
        }

        fn sample_predict(&self, w: &Self::ThetaLeaf) -> Self::Prediction {
            *w
        }
    }

    impl<'a, X> TrainingData<Sample<'a, X, f64>> for [Sample<'a, X, f64>]
        where X: Clone + PartialOrd + SampleRange + Bounded
    {
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

        fn partition_data(&mut self, split: &Split<usize, X>) -> (&mut Self, &mut Self) {
            let i = self.partition(|sample| sample.sample_as_split_feature(&split.theta) <= split.threshold);
            self.split_at_mut(i)
        }

        fn split_criterion(&self) -> f64 {
            let mean = f64::mean(self.iter().map(|sample| &sample.y));
            self.iter().map(|sample| sample.y - mean).map(|ym| ym * ym).sum::<f64>() / self.len() as f64
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

    impl<'a, X> AsTrainingData<Sample<'a, X, f64>> for Vec<Sample<'a, X, f64>>
        where X: Clone + PartialOrd + SampleRange + Bounded
    {
        type Description = [Sample<'a, X, f64>];
        fn as_training(&mut self) -> &mut Self::Description {
            &mut self[..]
        }
    }

    pub struct ExtraTreesRegressor {
        n_estimators: usize,
        n_splits: usize,
        min_samples_split: usize,
    }

    impl ExtraTreesRegressor {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn n_estimators(mut self, n: usize) -> Self {
            self.n_estimators = n;
            self
        }

        pub fn n_splits(mut self, n: usize) -> Self {
            self.n_splits = n;
            self
        }

        pub fn min_samples_split(mut self, n: usize) -> Self {
            self.min_samples_split = n;
            self
        }

        pub fn fit<'a, 'b, T>(&'a self, x: &'b Vec2D<T>, y: &'b Vec<f64>) -> DeterministicForest<Sample<'b, T, f64>>
            where T: Clone + cmp::PartialOrd + SampleRange + Bounded,
        {
            let mut data: Vec<Sample<T, f64>> = x.iter()
                .zip(y.iter())
                .map(|(xi, yi)| Sample{x: xi, y: *yi})
                .collect();

            DeterministicForestBuilder::new(
                self.n_estimators,
                DeterministicTreeBuilder::new(
                    self.min_samples_split,
                    BestRandomSplit::new(self.n_splits)
                )
            ).fit(&mut data)
        }
    }

    impl Default for ExtraTreesRegressor {
        fn default() -> Self {
            Self {
                n_estimators: 10,
                n_splits: 1,
                min_samples_split: 2,
            }
        }
    }
}


pub mod extra_trees_classifier {
    use super::*;

    pub type Builder<X> = EnsembleBuilder<CategoricalProbabilities, Data<X>, TreeBuilder<X>, Tree<X>>;

    pub type Model<X> = Ensemble<X, CategoricalProbabilities, Tree<X>>;

    pub type TreeBuilder<X> = DeterministicTreeBuilder<SplitFitter<X>, Predictor<X>>;

    pub type Tree<X> = DeterministicTree<Splitter<X>, Predictor<X>>;

    pub type Data<X> = [Sample<X>];
    pub type Sample<X> = TupleSample<Features, X, Y>;
    pub type Y = u8;

    pub type SplitFitter<X> = BestRandomSplit<Splitter<X>, SplitCriterion<X>, ThreadRng>;
    pub type Splitter<X> = ThresholdSplitter<Data<X>>;
    pub type Predictor<X> =  ClassPredictor<Sample<X>>;
    pub type Features = ColumnSelect;
    pub type SplitCriterion<X> = GiniCriterion<Sample<X>>;

    pub struct ExtraTreesClassifier {
        n_estimators: usize,
        n_splits: usize,
        min_samples_split: usize,
    }

    impl ExtraTreesClassifier {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn n_estimators(mut self, n: usize) -> Self {
            self.n_estimators = n;
            self
        }

        pub fn n_splits(mut self, n: usize) -> Self {
            self.n_splits = n;
            self
        }

        pub fn min_samples_split(mut self, n: usize) -> Self {
            self.min_samples_split = n;
            self
        }
    }

    impl Default for ExtraTreesClassifier {
        fn default() -> Self {
            Self {
                n_estimators: 10,
                n_splits: 1,
                min_samples_split: 2,
            }
        }
    }

    impl<X> LearnerMut<Data<X>, Model<X>> for ExtraTreesClassifier
        where X: Clone + GetItem,
              X::Item: Clone + cmp::PartialOrd + SampleRange,
    {

        fn fit(&self, data: &mut Data<X>) -> Model<X>
        {
            Builder::new(
                self.n_estimators,
                TreeBuilder::new(
                    SplitFitter::new(
                        self.n_splits,
                        thread_rng()),
                    self.min_samples_split,
                )
            ).fit(data)
        }
    }
}


#[cfg(test)]
mod tests {

    #[test]
    fn extra_trees_regressor() {
        use super::extra_trees_regressor::ExtraTreesRegressor;
        use vec2d::Vec2D;

        let x = Vec2D::from_slice(&[1, 2, 3,    7, 8, 9], 1);
        let y = vec![5.0, 5.0, 5.0,    2.0, 2.0, 2.0];

        let model = ExtraTreesRegressor::new()
            .n_estimators(2)
            .n_splits(1)
            .min_samples_split(2)
            .fit(&x, &y);

        assert_eq!(model.predict(&[-1000] as &[_]), 5.0);
        assert_eq!(model.predict(&[1000] as &[_]), 2.0);

        let p = model.predict(&[5] as &[_]);
        assert!(p >= 2.0);
        assert!(p <= 5.0);
    }

    #[test]
    fn extra_trees_classifier() {
        use super::extra_trees_classifier::ExtraTreesClassifier;
        use super::extra_trees_classifier::Sample;
        use LearnerMut;
        use Predictor as PT;

        let x = vec![[1], [2], [3],    [7], [8], [9]];
        let y = vec![ 1,   1,   1,      2,   2,   2];

        let mut data: Vec<Sample<[i32;1]>> = x.into_iter().zip(y.into_iter()).map(|(x, y)| Sample::new(x, y)).collect();

        let model = ExtraTreesClassifier::new()
            .n_estimators(100)
            .n_splits(1)
            .min_samples_split(2)
            .fit(&mut data);

        assert_eq!(model.predict(&[-1000]).prob(1), 1.0);
        assert_eq!(model.predict(&[1000]).prob(2), 1.0);

        let p = model.predict(&[5]);
        assert_eq!(p.prob(0), 0.0);
        assert!(p.prob(1) > 0.0);
        assert!(p.prob(2) > 0.0);
    }
}
