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

    fn from_usize(id: usize) -> Self{
        match id {
            0 => Classes::Red,
            1 => Classes::Green,
            2 => Classes::Blue,
            _ => panic!("Invalid class")
        }
    }

    fn n_categories(&self) -> Option<usize> {
        Some(3)
    }
}

/// Counter optimized for three classes
#[derive(Debug, Clone)]
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
        let a = self.p[0];
        let b = self.p[1];
        let c = self.p[2];

        if a > b {
            if a > c {
                T::from_usize(0)
            } else {
                T::from_usize(2)
            }
        } else {
            if b > c {
                T::from_usize(1)
            } else {
                T::from_usize(2)
            }
        }
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
