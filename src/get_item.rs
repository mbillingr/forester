pub trait GetItem {
    type Item;
    fn get_item(&self, i: usize) -> &Self::Item;
    fn n_items(&self) -> usize;
}

impl<T> GetItem for [T] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {self.len()}
}

impl<'a, T> GetItem for &'a [T] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {self.len()}
}

impl<T> GetItem for Vec<T> {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {self.len()}
}

macro_rules! impl_getitem_for_array {
    ( $( $size:expr ),* ) => {
        $(
            impl<T> GetItem for [T; $size] {
                type Item = T;
                fn get_item(&self, i: usize) -> &T {
                    &self[i]
                }
                fn n_items(&self) -> usize {$size}
            }
        )*
    };
}

impl_getitem_for_array!{ 1,  2,  3,  4,  5,  6,  7,  8}
impl_getitem_for_array!{ 9, 10, 11, 12, 13, 14, 15, 16}
impl_getitem_for_array!{17, 18, 19, 20, 21, 22, 23, 24}
impl_getitem_for_array!{25, 26, 27, 28, 29, 30, 31, 32}
