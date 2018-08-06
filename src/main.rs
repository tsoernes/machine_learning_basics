#![macro_use]
extern crate csv;
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate serde;
#[macro_use]
extern crate structopt;
extern crate num_traits;
extern crate rand;
#[macro_use]
extern crate clap;
extern crate quickersort;

mod decision_tree_regression;
mod k_nearest_neighbors;
mod logistic_regression;
mod utils;

use structopt::StructOpt;

type RngSeed = [u8; 32];
const RNG_SEED: RngSeed = [0; 32];

arg_enum! {
    #[derive(PartialEq, Debug)]
    pub enum Algo {
        DTR,
        KNN,
        LGR
    }
}

#[derive(StructOpt, Debug)]
#[structopt(name = "ML Basics")]
pub struct Opt {
    /// DTR: Decision Tree Regression. KNN: K Nearest Neighbors. LGR: Logistic Regression.
    #[structopt(raw(possible_values = "&Algo::variants()", case_insensitive = "true"))]
    algo: Algo,

    /// Decision tree: Max tree depth
    #[structopt(long = "max_depth", default_value = "4")]
    max_depth: usize,

    /// Decision tree: Minimum number of samples
    #[structopt(long = "min_samples", default_value = "2")]
    min_samples: usize,

    /// K Nearest Neighors: K
    #[structopt(short = "k", default_value = "5")]
    k: usize,

    /// Logistic regression: Learning rate
    #[structopt(long = "learning_rate", short = "a", default_value = "0.009")]
    learning_rate: f64,

    /// Logistic regression: Number of training iterations
    #[structopt(long = "n_iters", short = "n", default_value = "600")]
    n_iters: usize,

    /// Use random seed instead of a fixed seed for the RNG
    #[structopt(short = "r", long = "rand_rng")]
    rand_rng: bool,

    /// Train test data set split percent
    #[structopt(long = "tts", default_value = "0.75")]
    train_test_split_ratio: f64,
}

fn main() {
    let opt = Opt::from_args();
    let rng_seed = if opt.rand_rng { None } else { Some(RNG_SEED) };
    let tts = opt.train_test_split_ratio;
    match opt.algo {
        Algo::DTR => decision_tree_regression::run(opt.max_depth, opt.min_samples, tts, rng_seed),
        Algo::KNN => k_nearest_neighbors::run(opt.k, tts, rng_seed),
        Algo::LGR => logistic_regression::run(opt.n_iters, opt.learning_rate, tts, rng_seed),
    }
}
