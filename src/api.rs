
use criteria::VarCriterion;
use d_tree::{DeterministicTree, DeterministicTreeBuilder};
use datasets::TupleSample;
use ensemble::{Ensemble, EnsembleBuilder};
use features::ColumnSelect;
use get_item::GetItem;
use predictors::ConstMean;
use splitters::{BestRandomSplit, ThresholdSplitter};


pub mod ExtraTreesRegressor {
    use super::*;

    pub type Builder<X, Y> = EnsembleBuilder<Data<X, Y>, TreeBuilder<X, Y>, Tree<X, Y>>;

    pub type Model<X, Y> = Ensemble<X, Y, Tree<X, Y>>;

    pub type TreeBuilder<X, Y> = DeterministicTreeBuilder<SplitFitter<X, Y>, Predictor<X, Y>>;

    pub type Tree<X, Y> = DeterministicTree<Splitter<X, Y>, Predictor<X, Y>>;

    pub type Data<X, Y> = [Sample<X, Y>];
    pub type Sample<X, Y> = TupleSample<Features, X, Y>;

    type SplitFitter<X, Y> = BestRandomSplit<Splitter<X, Y>, SplitCriterion<X, Y>>;
    type Splitter<X, Y> = ThresholdSplitter<Data<X, Y>>;
    type Predictor<X, Y> =  ConstMean<Sample<X, Y>>;
    type Features = ColumnSelect;
    type SplitCriterion<X, Y> = VarCriterion<Sample<X, Y>>;

}


#[cfg(test)]
mod tests {

    #[test]
    fn extra_trees_regressor() {
        use super::ExtraTreesRegressor::*;
        use super::ExtraTreesRegressor::Builder;
        use super::ExtraTreesRegressor::Sample;
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
}
