//! Module for working with categorical data.

/// Mark a type as Categorical (for example class labels).
///
/// Each category is supposed to be convertible to a unique integer ID. IDs should start at 0 and
/// use continuous numbering. This is not strictly required but depending on implementations
/// may lead to performance improvements.
pub trait Categorical {
    /// Return unique id of present instance.
    fn as_usize(&self) -> usize;

    /// Create category from given id.
    fn from_usize(id: usize) -> Self;

    /// Total number of categories.
    ///
    /// This function can return `None` if the number is not known (e.g. number of classes depends
    /// on runtime data).
    fn n_categories(&self) -> Option<usize>;
}

/// Trait for counting instances of categorical variables.
pub trait CatCount<C>
    where C: Categorical
{
    /// Add a single observation of given category (increase this category's count by 1).
    fn add(&mut self, c: C);

    /// Add `n` observations of given category (increase this category's count by n).
    fn add_n(&mut self, c: C, n: usize);

    /// Remove a single observation of given category (decrease this category's count by 1)
    fn remove(&mut self, _c: C) { unimplemented!() }

    /// Return probability (relative count) of given category.
    fn probability(&self, c: C) -> f64;

    /// Return the most frequent category.
    fn most_frequent(&self) -> C;
}


impl Categorical for u8 {
    fn as_usize(&self) -> usize {
        *self as usize
    }

    fn from_usize(id: usize) -> Self {
        id as u8
    }

    fn n_categories(&self) -> Option<usize> {
        None
    }
}

impl Categorical for usize {
    fn as_usize(&self) -> usize {
        *self
    }

    fn from_usize(id: usize) -> Self {
        id
    }

    fn n_categories(&self) -> Option<usize> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct GenericCatCounter {
    counts: Vec<usize>,
    total: usize,
}

impl GenericCatCounter {
    pub fn new() -> Self {
        GenericCatCounter {
            counts: Vec::new(),
            total: 0,
        }
    }

    pub fn probs<F: FnMut(f64)>(&self, mut f: F) {
        let n = self.total as f64;
        for c in self.counts.iter() {
            f(*c as f64 / n);
        }
    }
}

impl<C: Categorical> CatCount<C> for GenericCatCounter {
    fn add(&mut self, c: C) {
        self.add_n(c, 1)
    }

    fn add_n(&mut self, c: C, n: usize) {
        let i = c.as_usize();
        if i >= self.counts.len() {
            self.counts.resize(i + 1, 0);
        }
        self.counts[i] += n;
        self.total += n;
    }

    fn remove(&mut self, c: C) {
        let i = c.as_usize();
        debug_assert!(i < self.counts.len());
        debug_assert!(self.counts[i] > 0);
        debug_assert!(self.total > 0);
        self.counts[i] -= 1;
        self.total -= 1;
    }

    fn probability(&self, c: C) -> f64 {
        let i = c.as_usize();
        if i < self.counts.len() {
            self.counts[i] as f64 / self.total as f64
        } else {
            0.0
        }
    }

    fn most_frequent(&self) -> C {
        let mut n = 0;
        let mut c = 0;
        for i in 0..self.counts.len() {
            // TODO: handle ties?
            if self.counts[i] > n {
                n = self.counts[i];
                c = i;
            }
        }
        C::from_usize(c)
    }
}


#[test]
fn cat_u8() {
    let x = u8::from_usize(42);
    assert_eq!(x, 42u8);
    assert_eq!(x.as_usize(), 42usize);
    assert_eq!(x.n_categories(), None);
}