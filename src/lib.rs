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

pub fn dummy1() {
  println!("The purpose of this function is to be not covered in any tests, and therefore to show up in the coverage report.");
}


#[cfg(test)]
mod tests {

  pub fn dummy2() {
    println!("Same here...");
    println!("The purpose of this function is to be not covered in any tests, and therefore to show up in the coverage report.");
  }
  
}
