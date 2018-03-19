pub trait GetItem {
    type Item;
    fn get_item(&self, i: usize) -> &Self::Item;
}

impl<T> GetItem for [T] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
}

impl<T> GetItem for Vec<T> {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
}

// TODO: Implement GetItem for many more array sizes... (macro!)

impl<T> GetItem for [T; 0] {
    type Item = T;
    fn get_item(&self, _: usize) -> &T {
        panic!("Getting item from zero-length array")
    }
}

impl<T> GetItem for [T; 1] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
}

impl<T> GetItem for [T; 2] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
}

impl<T> GetItem for [T; 3] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
}
