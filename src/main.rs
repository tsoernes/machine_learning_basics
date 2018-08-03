#![macro_use]
extern crate csv;
extern crate ndarray;
extern crate serde;
#[macro_use]
extern crate structopt;
extern crate num_traits;
extern crate rand;

mod decision_tree_regression;
mod k_nearest_neighbors;
mod utils;

use structopt::StructOpt;

const RNG_SEED: [u8; 32] = [0; 32];
#[derive(StructOpt, Debug)]
#[structopt(name = "ML Basics")]
pub struct Opt {
    /// Decision tree: Max tree depth
    #[structopt(long = "max_depth", default_value = "4")]
    max_depth: usize,

    /// Decision tree: Minimum number of samples
    #[structopt(long = "min_samples", default_value = "2")]
    min_samples: usize,

    /// Use random seed instead of a fixed seed for the RNG
    #[structopt(short = "r", long = "rand_rng")]
    rand_rng: bool,
}

fn main() {
    let opt = Opt::from_args();
    let rng_seed = if opt.rand_rng { None } else { Some(RNG_SEED) };
    decision_tree_regression::run(opt.max_depth, opt.min_samples, rng_seed);
}
