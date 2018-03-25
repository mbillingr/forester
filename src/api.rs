
use rand::ThreadRng;

use criteria::VarCriterion;
use d_tree::{DeterministicTree, DeterministicTreeBuilder};
use datasets::TupleSample;
use ensemble::{Ensemble, EnsembleBuilder};
use features::ColumnSelect;
use get_item::GetItem;
use predictors::{CategoricalProbabilities, ClassPredictor, ConstMean};
use splitters::{BestRandomSplit, ThresholdSplitter};


pub mod extra_trees_regressor {
    use super::*;

    pub type Builder<X, Y> = EnsembleBuilder<Y, Data<X, Y>, TreeBuilder<X, Y>, Tree<X, Y>>;

    pub type Model<X, Y> = Ensemble<X, Y, Tree<X, Y>>;

    pub type TreeBuilder<X, Y> = DeterministicTreeBuilder<SplitFitter<X, Y>, Predictor<X, Y>>;

    pub type Tree<X, Y> = DeterministicTree<Splitter<X, Y>, Predictor<X, Y>>;

    pub type Data<X, Y> = [Sample<X, Y>];
    pub type Sample<X, Y> = TupleSample<Features, X, Y>;

    type SplitFitter<X, Y> = BestRandomSplit<Splitter<X, Y>, SplitCriterion<X, Y>, ThreadRng>;
    type Splitter<X, Y> = ThresholdSplitter<Data<X, Y>>;
    type Predictor<X, Y> =  ConstMean<Sample<X, Y>>;
    type Features = ColumnSelect;
    type SplitCriterion<X, Y> = VarCriterion<Sample<X, Y>>;
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

    type SplitFitter<X> = BestRandomSplit<Splitter<X>, SplitCriterion<X>, ThreadRng>;
    type Splitter<X> = ThresholdSplitter<Data<X>>;
    type Predictor<X> =  ClassPredictor<Sample<X>>;
    type Features = ColumnSelect;
    type SplitCriterion<X> = VarCriterion<Sample<X>>;
}


#[cfg(test)]
mod tests {

    #[test]
    fn extra_trees_regressor() {
        use super::extra_trees_regressor::*;
        use super::extra_trees_regressor::Builder;
        use super::extra_trees_regressor::Sample;
        use LearnerMut;
        use Predictor as PT;
        use Sample as SampleTrait;
        use get_item::GetItem;

        let x = vec![[1], [2], [3],    [7], [8], [9]];
        let y = vec![5.0, 5.0, 5.0,    2.0, 2.0, 2.0];

        let mut data: Vec<Sample<[i32;1], f64>> = x.into_iter().zip(y.into_iter()).map(|(x, y)| Sample::new(x, y)).collect();

        let model = Builder::default().fit(&mut data);
        let tree = TreeBuilder::default().fit(&mut data);

        assert_eq!(model.predict(&[-1000]), 5.0);
        assert_eq!(model.predict(&[1000]), 2.0);

        let p = model.predict(&[5]);
        assert!(p >= 2.0);
        assert!(p <= 5.0);
    }

    #[test]
    fn extra_trees_classifier() {
        use super::extra_trees_classifier::*;
        use super::extra_trees_classifier::Builder;
        use super::extra_trees_classifier::Sample;
        use LearnerMut;
        use Predictor as PT;
        use Sample as SampleTrait;
        use get_item::GetItem;

        let x = vec![[1], [2], [3],    [7], [8], [9]];
        let y = vec![ 1,   1,   1,      2,   2,   2];

        let mut data: Vec<Sample<[i32;1]>> = x.into_iter().zip(y.into_iter()).map(|(x, y)| Sample::new(x, y)).collect();

        let model = Builder::default().fit(&mut data);
        let tree = TreeBuilder::default().fit(&mut data);

        assert_eq!(model.predict(&[-1000]).prob(1), 1.0);
        assert_eq!(model.predict(&[1000]).prob(2), 1.0);

        println!("{:?}", model.predict(&[5]));


        let p = model.predict(&[5]);
        assert_eq!(p.prob(0), 0.0);
        assert!(p.prob(1) > 0.0);
        assert!(p.prob(2) > 0.0);
    }
}
