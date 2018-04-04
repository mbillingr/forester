
use std::marker::PhantomData;

use super::SplitCriterion;
use super::Real;
use super::Sample;

use predictors::CategoricalProbabilities;


pub struct VarCriterion<D: ?Sized> {
    _p: PhantomData<D>
}

impl<S> SplitCriterion<S> for VarCriterion<S>
    where S: Sample,
          S::Y: Copy + Into<Real>,
{
    type C = Real;

    fn calc_presplit(data: &[S]) -> Real {
        let mut sum = 0.0;
        let mut ssum = 0.0;

        for sample in data {
            let yi: Real = (*sample.get_y()).into();
            sum += yi;
            ssum += yi * yi;
        }

        let n = data.len() as Real;

        let mean = sum / n;
        ssum / n - mean * mean
    }

    fn calc_postsplit(yl: &[S], yr: &[S]) -> Real {
        let a = yl.len() as Real;
        let b = yr.len() as Real;
        (Self::calc_presplit(yl) * a + Self::calc_presplit(yr) * b) / (a + b)
    }
}

pub struct GiniCriterion<D: ?Sized> {
    _p: PhantomData<D>
}

impl<S> SplitCriterion<S> for GiniCriterion<S>
    where S: Sample<Y=u8>
{
    type C = Real;

    fn calc_presplit(data: &[S]) -> Real {
        let mut counts = CategoricalProbabilities::new();

        for sample in data {
            let yi = *sample.get_y();
            counts.add_one(yi);
        }

        let mut gini = 0.0;
        counts.apply(|p| gini += p*(1.0-p));

        gini
    }

    fn calc_postsplit(yl: &[S], yr: &[S]) -> Real {
        let a = yl.len() as Real;
        let b = yr.len() as Real;
        (Self::calc_presplit(yl) * a + Self::calc_presplit(yr) * b) / (a + b)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use features::ColumnSelect;
    use datasets::TupleSample;

    #[test]
    fn var_criterion() {
        let data: Vec<TupleSample<ColumnSelect, [();0], _>>;
        data = vec![TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 11),
                    TupleSample::new([], 12),
                    TupleSample::new([], 11),
                    TupleSample::new([], 12)];

        let c = VarCriterion::calc_presplit(&data);
        assert_eq!(c, 25.25);

        let c = VarCriterion::calc_postsplit(&data[..4], &data[4..]);
        assert_eq!(c, 0.25);

        let data: Vec<TupleSample<ColumnSelect, [();0], _>>;
        data = vec![TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 11),
                    TupleSample::new([], 12)];

        let c = VarCriterion::calc_postsplit(&data[..6], &data[6..]);
        assert_eq!(c, 0.25);
    }

    #[test]
    fn gini_criterion() {
        let data: Vec<TupleSample<ColumnSelect, [();0], _>>;
        data = vec![TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 3),
                    TupleSample::new([], 4)];

        let c = GiniCriterion::calc_presplit(&data);
        assert_eq!(c, 0.6875);

        let c = GiniCriterion::calc_postsplit(&data[..4], &data[4..]);
        assert_eq!(c, 0.625);

        let data: Vec<TupleSample<ColumnSelect, [();0], _>>;
        data = vec![TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 1),
                    TupleSample::new([], 2),
                    TupleSample::new([], 3),
                    TupleSample::new([], 4)];

        let c = GiniCriterion::calc_postsplit(&data[..6], &data[6..]);
        assert_eq!(c, 0.5);
    }
}
