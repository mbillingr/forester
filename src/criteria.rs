
use std::marker::PhantomData;

use super::DataSet;
use super::Float;
use super::FixedLength;
use super::Sample;
use super::SplitCriterion;


struct VarCriterion<D: ?Sized> {
    _p: PhantomData<D>
}

impl<S> SplitCriterion for VarCriterion<S>
    where S: Sample,
          S::Y: Copy + Into<f64>,
{
    type D = [S];
    type C = f64;

    fn calc_presplit(data: &Self::D) -> f64 {
        let mut sum = 0.0;
        let mut ssum = 0.0;

        for sample in data {
            let yi: f64 = sample.get_y().into();
            sum += yi;
            ssum += yi * yi;
        }

        let n = data.len() as f64;

        let mean = sum / n;

        ssum / n - mean * mean

        //let mean: f64 = y.into_iter().map(|yi| (*yi).into()).sum::<f64>() / y.len() as f64;
        //y.into_iter().map(|yi| (*yi).into() ).map(|yi| yi * yi).sum::<f64>() / y.len() as f64 - mean * mean
    }

    fn calc_postsplit(yl: &Self::D, yr: &Self::D) -> f64 {
        let a = yl.len() as f64;
        let b = yr.len() as f64;
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
