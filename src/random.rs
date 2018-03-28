
use rand::{thread_rng, Rng, ThreadRng};

/// Default method to initialize a Rng
pub trait DefaultRng : Rng {
    fn default_rng() -> Self;
}


impl DefaultRng for ThreadRng {
    fn default_rng() -> Self {
        thread_rng()
    }
}