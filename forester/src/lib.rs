extern crate rand;

pub mod api;
pub mod array_ops;
pub mod criteria;
pub mod datasets;
pub mod features;
pub mod ensemble;
pub mod get_item;
pub mod predictors;
pub mod random;
pub mod splitters;
pub mod traits;
pub mod d_tree;
//pub mod vec2d;

type Real = f64;

pub use traits::*;


#[cfg(test)]
mod tests {
}
