use std::iter::Sum;

pub trait IterMean<A=Self>
{
    fn mean<I: ExactSizeIterator<Item=A>>(iter: I) -> Self;
}

macro_rules! impl_mean {
    ($a:ident -> $b:ident) => (
        impl IterMean<$a> for $b {
            fn mean<I: ExactSizeIterator<Item=$a>>(iter: I) -> $b {
                let n = iter.len() as $b;
                let sum: $a = iter.sum();
                sum as $b / n
            }
        }

        impl<'a> IterMean<&'a $a> for $b {
            fn mean<I: ExactSizeIterator<Item=&'a $a>>(iter: I) -> $b {
                let n = iter.len() as $b;
                let sum: $a = iter.sum();
                sum as $b / n
            }
        }
    )
}

impl_mean! { f32 -> f32 }
impl_mean! { f64 -> f32 }
impl_mean! { i8 -> f32 }
impl_mean! { u8 -> f32 }
impl_mean! { i16 -> f32 }
impl_mean! { u16 -> f32 }
impl_mean! { i32 -> f32 }
impl_mean! { u32 -> f32 }
impl_mean! { i64 -> f32 }
impl_mean! { u64 -> f32 }
impl_mean! { isize -> f32 }
impl_mean! { usize -> f32 }

impl_mean! { f32 -> f64 }
impl_mean! { f64 -> f64 }
impl_mean! { i8 -> f64 }
impl_mean! { u8 -> f64 }
impl_mean! { i16 -> f64 }
impl_mean! { u16 -> f64 }
impl_mean! { i32 -> f64 }
impl_mean! { u32 -> f64 }
impl_mean! { i64 -> f64 }
impl_mean! { u64 -> f64 }
impl_mean! { isize -> f64 }
impl_mean! { usize -> f64 }


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iter_mean() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(f32::mean(x.iter()), 2.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 2.5);
        assert_eq!(f64::mean(x.iter()), 2.5);
        assert_eq!(f64::mean(x.into_iter()), 2.5);

        let x = vec![1.0f64, 2.0, 3.0, 4.0];
        assert_eq!(f32::mean(x.iter()), 2.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 2.5);
        assert_eq!(f64::mean(x.iter()), 2.5);
        assert_eq!(f64::mean(x.into_iter()), 2.5);

        let x = vec![1i8, 2, 3, -4];
        assert_eq!(f32::mean(x.iter()), 0.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 0.5);
        assert_eq!(f64::mean(x.iter()), 0.5);
        assert_eq!(f64::mean(x.into_iter()), 0.5);

        let x = vec![1u8, 2, 3, 4];
        assert_eq!(f32::mean(x.iter()), 2.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 2.5);
        assert_eq!(f64::mean(x.iter()), 2.5);
        assert_eq!(f64::mean(x.into_iter()), 2.5);

        let x = vec![1i16, 2, 3, -4];
        assert_eq!(f32::mean(x.iter()), 0.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 0.5);
        assert_eq!(f64::mean(x.iter()), 0.5);
        assert_eq!(f64::mean(x.into_iter()), 0.5);

        let x = vec![1u16, 2, 3, 4];
        assert_eq!(f32::mean(x.iter()), 2.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 2.5);
        assert_eq!(f64::mean(x.iter()), 2.5);
        assert_eq!(f64::mean(x.into_iter()), 2.5);

        let x = vec![1i32, 2, 3, -4];
        assert_eq!(f32::mean(x.iter()), 0.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 0.5);
        assert_eq!(f64::mean(x.iter()), 0.5);
        assert_eq!(f64::mean(x.into_iter()), 0.5);

        let x = vec![1u32, 2, 3, 4];
        assert_eq!(f32::mean(x.iter()), 2.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 2.5);
        assert_eq!(f64::mean(x.iter()), 2.5);
        assert_eq!(f64::mean(x.into_iter()), 2.5);

        let x = vec![1i64, 2, 3, -4];
        assert_eq!(f32::mean(x.iter()), 0.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 0.5);
        assert_eq!(f64::mean(x.iter()), 0.5);
        assert_eq!(f64::mean(x.into_iter()), 0.5);

        let x = vec![1u64, 2, 3, 4];
        assert_eq!(f32::mean(x.iter()), 2.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 2.5);
        assert_eq!(f64::mean(x.iter()), 2.5);
        assert_eq!(f64::mean(x.into_iter()), 2.5);

        let x = vec![1isize, 2, 3, -4];
        assert_eq!(f32::mean(x.iter()), 0.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 0.5);
        assert_eq!(f64::mean(x.iter()), 0.5);
        assert_eq!(f64::mean(x.into_iter()), 0.5);

        let x = vec![1usize, 2, 3, 4];
        assert_eq!(f32::mean(x.iter()), 2.5);
        assert_eq!(f32::mean(x.clone().into_iter()), 2.5);
        assert_eq!(f64::mean(x.iter()), 2.5);
        assert_eq!(f64::mean(x.into_iter()), 2.5);
    }
}
