
/// Simple trait, that determines where to put a split between two feature values.
///
/// Splits are supposed to always use <= for comparison. Thus, `split_between` should round up to
/// the next higher valid value. E.g.:
///
///   3, 4 -> 4
///   3, 5 -> 4
///   3, 6 -> 5
///
///   3.0, 4.0 -> 3.5
///   3.0, 5.0 -> 4.0
///   3.0, 6.0 -> 4.5
pub trait SplitBetween {
    fn split_between(&self, other: &Self) -> Self;
}

impl SplitBetween for f64 {
    fn split_between(&self, other: &Self) -> Self {
        (self + other) / 2.0
    }
}

impl SplitBetween for f32 {
    fn split_between(&self, other: &Self) -> Self {
        (self + other) / 2.0
    }
}

impl SplitBetween for i64 {
    fn split_between(&self, other: &Self) -> Self {
        let diff = other - self;
        if diff % 2 == 0 {
            self + diff / 2
        } else {
            self + diff / 2 + 1
        }
    }
}

impl SplitBetween for u64 {
    fn split_between(&self, other: &Self) -> Self {
        let diff = other - self;
        if diff % 2 == 0 {
            self + diff / 2
        } else {
            self + diff / 2 + 1
        }
    }
}

impl SplitBetween for isize {
    fn split_between(&self, other: &Self) -> Self {
        let diff = other - self;
        if diff % 2 == 0 {
            self + diff / 2
        } else {
            self + diff / 2 + 1
        }
    }
}

impl SplitBetween for usize {
    fn split_between(&self, other: &Self) -> Self {
        let diff = other - self;
        if diff % 2 == 0 {
            self + diff / 2
        } else {
            self + diff / 2 + 1
        }
    }
}

impl SplitBetween for i32 {
    fn split_between(&self, other: &Self) -> Self {
        let diff = other - self;
        if diff % 2 == 0 {
            self + diff / 2
        } else {
            self + diff / 2 + 1
        }
    }
}

impl SplitBetween for u32 {
    fn split_between(&self, other: &Self) -> Self {
        let diff = other - self;
        if diff % 2 == 0 {
            self + diff / 2
        } else {
            self + diff / 2 + 1
        }
    }
}

impl SplitBetween for i16 {
    fn split_between(&self, other: &Self) -> Self {
        let diff = other - self;
        if diff % 2 == 0 {
            self + diff / 2
        } else {
            self + diff / 2 + 1
        }
    }
}

impl SplitBetween for u16 {
    fn split_between(&self, other: &Self) -> Self {
        let diff = other - self;
        if diff % 2 == 0 {
            self + diff / 2
        } else {
            self + diff / 2 + 1
        }
    }
}

impl SplitBetween for i8 {
    fn split_between(&self, other: &Self) -> Self {
        let diff = other - self;
        if diff % 2 == 0 {
            self + diff / 2
        } else {
            self + diff / 2 + 1
        }
    }
}

impl SplitBetween for u8 {
    fn split_between(&self, other: &Self) -> Self {
        let diff = other - self;
        if diff % 2 == 0 {
            self + diff / 2
        } else {
            self + diff / 2 + 1
        }
    }
}