
use std::marker::PhantomData;

use super::SplitCriterion;
use super::Real;
use super::Sample;


pub struct VarCriterion<D: ?Sized> {
    _p: PhantomData<D>
}

impl<S> SplitCriterion for VarCriterion<S>
    where S: Sample,
          S::Y: Copy + Into<Real>,
{
    type D = [S];
    type C = Real;

    fn calc_presplit(data: &Self::D) -> Real {
        let mut sum = 0.0;
        let mut ssum = 0.0;

        for sample in data {
            let yi: Real = sample.get_y().into();
            sum += yi;
            ssum += yi * yi;
        }

        let n = data.len() as Real;

        let mean = sum / n;
        ssum / n - mean * mean
    }

    fn calc_postsplit(yl: &Self::D, yr: &Self::D) -> Real {
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
}
