//! Label definitions for a three-class classification problem.

use std::iter::Sum;

use forester::categorical::{Categorical, CatCount};
use forester::iter_mean::IterMean;

/// Three classes, labelled `Red`, `Green`, `Blue`.
#[derive(Copy, Clone)]
pub enum Classes {
    Red,
    Green,
    Blue,
}

impl Categorical for Classes {
    fn as_usize(&self) -> usize {
        *self as usize
    }

    fn n_categories(&self) -> Option<usize> {
        Some(3)
    }
}

/// Counter optimized for three classes
#[derive(Clone)]
pub struct ClassCounts {
    p: [usize; 3],
}

impl ClassCounts {
    fn new() -> Self {
        ClassCounts {
            p: [0; 3]
        }
    }
}

impl CatCount<Classes> for ClassCounts {
    fn add(&mut self, c: Classes) {
        self.p[c as usize] += 1;
    }

    fn add_n(&mut self, c: Classes, n: usize) {
        self.p[c as usize] += n;
    }

    fn probability(&self, c: Classes) -> f64 {
        self.p[c as usize] as f64 / self.p.iter().sum::<usize>() as f64
    }
}

impl<'a> Sum<&'a Classes> for ClassCounts {
    fn sum<I: Iterator<Item=&'a Classes>>(iter: I) -> Self {
        let mut counts = ClassCounts::new();
        for c in iter {
            counts.add(*c);
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
