#![macro_use]
extern crate csv;
extern crate ndarray;
extern crate serde;
#[macro_use]
extern crate structopt;
extern crate rand;

mod decision_tree_regression;

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
}

fn main() {
    let opt = Opt::from_args();
    decision_tree_regression::run(opt.max_depth, opt.min_samples);
}
