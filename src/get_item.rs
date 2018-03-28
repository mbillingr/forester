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

// TODO: Implement GetItem for many more array sizes... (macro!)

impl<T> GetItem for [T; 0] {
    type Item = T;
    fn get_item(&self, _: usize) -> &T {
        panic!("Getting item from zero-length array")
    }
    fn n_items(&self) -> usize {0}
}

impl<T> GetItem for [T; 1] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {1}
}

impl<T> GetItem for [T; 2] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {2}
}

impl<T> GetItem for [T; 3] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {3}
}

impl<T> GetItem for [T; 4] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {3}
}

impl<T> GetItem for [T; 5] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {3}
}

impl<T> GetItem for [T; 10] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {3}
}

impl<T> GetItem for [T; 20] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {3}
}

impl<T> GetItem for [T; 30] {
    type Item = T;
    fn get_item(&self, i: usize) -> &T {
        &self[i]
    }
    fn n_items(&self) -> usize {3}
}
