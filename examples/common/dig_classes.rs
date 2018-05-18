//! Label definitions for a three-class classification problem.

use std::iter::Sum;

use forester::categorical::{Categorical, CatCount};
use forester::iter_mean::IterMean;

/// Three classes, labelled `Red`, `Green`, `Blue`.
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Digit(pub u8);

impl Categorical for Digit {
    fn as_usize(&self) -> usize {
        self.0 as usize
    }

    fn from_usize(id: usize) -> Self{
        if id > 255 {
            panic!("Invalid class")
        }
        Digit(id as u8)
    }

    fn n_categories(&self) -> Option<usize> {
        Some(10)
    }
}

/// Counter optimized for three classes
#[derive(Debug, Clone)]
pub struct ClassCounts {
    p: [usize; 10],
}

impl ClassCounts {
    fn new() -> Self {
        ClassCounts {
            p: [0; 10]
        }
    }
}

impl<T> CatCount<T> for ClassCounts
    where T: Categorical
{
    fn add(&mut self, c: T) {
        self.p[c.as_usize()] += 1;
    }

    fn add_n(&mut self, c: T, n: usize) {
        self.p[c.as_usize()] += n;
    }

    fn probability(&self, c: T) -> f64 {
        self.p[c.as_usize()] as f64 / self.p.iter().sum::<usize>() as f64
    }

    fn most_frequent(&self) -> T {
        let i = self.p
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .unwrap()
            .0;

        T::from_usize(i)
    }
}

impl<T> Sum<T> for ClassCounts
    where T: Categorical
{
    fn sum<I: Iterator<Item=T>>(iter: I) -> Self {
        let mut counts = ClassCounts::new();
        for c in iter {
            counts.add(c);
        }
        counts
    }
}

impl IterMean<ClassCounts> for ClassCounts {
    fn mean<I: ExactSizeIterator<Item=ClassCounts>>(iter: I) -> Self {
        let mut total_counts = ClassCounts::new();
        for c in iter {
            for (a, b) in total_counts.p.iter_mut().zip(c.p.iter()) {
                *a += *b;
            }
        }
        total_counts
    }
}
