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

    /// Return probability (relative count) of given category.
    fn probability(&self, c: C) -> f64;

    /// Return the most frequent category.
    fn most_frequent(&self) -> C;
}
