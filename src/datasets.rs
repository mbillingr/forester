use std::cmp;
use std::marker::PhantomData;

use super::Feature;
use super::Sample;

use get_item::GetItem;

#[derive(Debug)]
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
    type FX = FX;
    type X = X;
    type Y = Y;

    /*fn get_feature(&self, theta: &FX::Theta) -> Self::F {
        FX::get_feature(&self.data.0, theta)
    }*/

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
    use splitters::ThresholdSplitter;

    #[test]
    fn dataset_sort() {
        let s: TupleSample<Mix2, _, _> = TupleSample{data: ([1, 2], 3), _p: PhantomData};
        assert_eq!(Mix2::get_feature(&s.get_x(), &(0, 1, 0.5, 0.5)), 1.5);

        let data: &mut [TupleSample<ColumnSelect, _, _>] =
            &mut [TupleSample{data: ([0, 3], 1), _p: PhantomData},
                 TupleSample{data: ([2, 0], 2), _p: PhantomData},
                 TupleSample{data: ([1, 1], 3), _p: PhantomData},
                 TupleSample{data: ([3, 2], 4), _p: PhantomData}] as &mut [_];

        assert_eq!(ColumnSelect::get_feature(&data[0].get_x(), &1), 3);
        assert_eq!(data.n_samples(), 4);

        let i = data.partition_by_split(&ThresholdSplitter::new(1, 1));
        assert_eq!(i, 2);
        assert_eq!(data[0].get_y(), 2);
        assert_eq!(data[1].get_y(), 3);
        assert_eq!(data[2].get_y(), 1);
        assert_eq!(data[3].get_y(), 4);

        {
            let (a, _b) = data.split_at_mut(2);
            let i = a.partition_by_split(&ThresholdSplitter::new(0, 1));
            assert_eq!(i, 1);
        }

        assert_eq!(data[0].get_y(), 3);
        assert_eq!(data[1].get_y(), 2);
        assert_eq!(data[2].get_y(), 1);
        assert_eq!(data[3].get_y(), 4);

    }
}