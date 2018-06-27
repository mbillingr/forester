//! Module for working with continuous data.

/// Mark a type as Continuous (for example regression targets).
pub trait Continuous {
    /// Return unique id of present instance.
    fn as_float(&self) -> f64;

    /// Create category from given id.
    fn from_float(f: f64) -> Self;
}

impl Continuous for f64 {
    #[inline(always)]
    fn as_float(&self) -> f64 { *self }

    #[inline(always)]
    fn from_float(f: f64) -> f64 { f }
}

impl Continuous for f32 {
    #[inline(always)]
    fn as_float(&self) -> f64 { *self as f64}

    #[inline(always)]
    fn from_float(f: f64) -> f32 { f as f32 }
}
