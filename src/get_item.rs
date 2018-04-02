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

impl<T> GetItem for [T; 0] {
    type Item = T;
    fn get_item(&self, _: usize) -> &T {
        panic!("Getting item from zero-length array")
    }
    fn n_items(&self) -> usize {0}
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



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_getitem() {
        let x = [1, 2, 3];
        assert_eq!(x.get_item(1), &2);
        assert_eq!(x.n_items(), 3);

        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(x.get_item(3), &4.0);
        assert_eq!(x.n_items(), 5);
    }

    #[test]
    fn vec_getitem() {
        let x = vec![1, 2, 3];
        assert_eq!(x.get_item(1), &2);
        assert_eq!(x.n_items(), 3);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(x.get_item(3), &4.0);
        assert_eq!(x.n_items(), 5);
    }

    #[test]
    fn slice_getitem() {
        let y = vec![1, 2, 3];
        let x: &[_] = &y;
        assert_eq!(x.get_item(1), &2);
        assert_eq!(x.n_items(), 3);

        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x: &[_] = &y;
        assert_eq!(x.get_item(3), &4.0);
        assert_eq!(x.n_items(), 5);
    }

    #[test]
    #[should_panic(expected = "Getting item from zero-length array")]
    fn empty_getitem() {
        let x: [i32; 0] = [];
        assert_eq!(x.n_items(), 0);
        x.get_item(0);
    }
}
