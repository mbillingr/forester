use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::ops;
use std::slice;

use super::Feature;
use super::Sample;

use get_item::GetItem;

//use super::DataSet;
//use super::FeatureSet;
//use super::OutcomeVariable;



trait Sortable<T> {
    fn sort_unstable_by<F>(&mut self, predicate: F)
        where F: FnMut(&T, &T) -> cmp::Ordering;
}


pub struct TupleSample<FX, X, Y> {
    data: (X, Y),
    _p: PhantomData<FX>,
}

impl<FX, X, Y> TupleSample<FX, X, Y> {
    pub fn new(x: X, y: Y) -> Self {
        TupleSample {
            data: (x, y),
            _p: PhantomData,
        }
    }
}

impl<X, Y, FX> Sample for TupleSample<FX, X, Y>
    where X: Clone + GetItem,
          Y: Clone,
          FX: Feature<X>,
          FX::F: Clone + cmp::PartialOrd,
{
    type Theta = FX::Theta;
    type F = FX::F;
    type X = X;
    type Y = Y;

    fn get_feature(&self, theta: &FX::Theta) -> Self::F {
        FX::get_feature(&self.data.0, theta)
    }

    fn get_x(&self) -> Self::X {
        self.data.0.clone()
    }

    fn get_y(&self) -> Self::Y {
        self.data.1.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;
    use super::*;
    use features::{ColumnSelect, Mix2};
    use super::super::DataSet;

    #[test]
    fn dataset_sort() {
        let s: TupleSample<Mix2, _, _> = TupleSample{data: ([1, 2], 3), _p: PhantomData};
        assert_eq!(s.get_feature(&(0, 1, 0.5)), 1.5);

        let data: &mut [TupleSample<ColumnSelect, _, _>] =
            &mut [TupleSample{data: ([0, 3], 1), _p: PhantomData},
                 TupleSample{data: ([2, 0], 2), _p: PhantomData},
                 TupleSample{data: ([1, 1], 3), _p: PhantomData},
                 TupleSample{data: ([3, 2], 4), _p: PhantomData}] as &mut [_];

        assert_eq!(data[0].get_feature(&1), 3);
        assert_eq!(data.n_samples(), 4);

        data.sort_by_feature(&1);
        assert_eq!(data[0].get_y(), 2);
        assert_eq!(data[1].get_y(), 3);
        assert_eq!(data[2].get_y(), 4);
        assert_eq!(data[3].get_y(), 1);

        {
            let (a, _b) = data.split_at_mut(2);
            a.sort_by_feature(&0);
        }

        assert_eq!(data[0].get_y(), 3);
        assert_eq!(data[1].get_y(), 2);
        assert_eq!(data[2].get_y(), 4);
        assert_eq!(data[3].get_y(), 1);

    }
}