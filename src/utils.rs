use ndarray::*;
use num_traits::identities::Zero;
use rand::{thread_rng, ChaChaRng, Rng, SeedableRng};
use std::fmt::Debug;
use std::str::FromStr;

#[derive(Debug)]
pub struct Dataset<X, Y> {
    pub x_train: Array2<X>,
    pub y_train: Array1<Y>,
    pub x_test: Array2<X>,
    pub y_test: Array1<Y>,
}

/// Shuffle the data set, then split into training and test set with ratio 'train_test_split'.
/// If rng_seed is given, then use it for shuffling.
pub fn shuffle_split<
    X: Clone + Copy + FromStr + Zero + Debug,
    Y: Clone + Copy + FromStr + Zero + Debug,
>(
    data_x: Array2<X>,
    data_y: Array1<Y>,
    train_test_split: f64,
    rng_seed: Option<[u8; 32]>,
) -> Dataset<X, Y> {
    let n_samples = data_x.rows();
    let n_features = data_x.cols();

    // Shuffle the data set
    let mut indecies: Vec<usize> = (0..n_samples).collect();
    match rng_seed {
        Some(seed) => {
            ChaChaRng::from_seed(seed).shuffle(&mut indecies);
        }
        _ => {
            thread_rng().shuffle(&mut indecies);
        }
    };
    let data_x = data_x.select(Axis(0), &indecies);
    let data_y = data_y.select(Axis(0), &indecies);

    // Split data set into test and training set.
    let n_train = (train_test_split * n_samples as f64) as usize;
    let n_test = n_samples - n_train;
    let mut x_train: Array2<X> = Array::zeros((n_train, n_features));
    let mut y_train: Array1<Y> = Array::zeros(n_train);
    let mut x_test: Array2<X> = Array::zeros((n_test, n_features));
    let mut y_test: Array1<Y> = Array::zeros(n_test);
    for (i, (x, y)) in data_x.outer_iter().zip(data_y.into_iter()).enumerate() {
        if i < n_train {
            x_train.slice_mut(s![i, ..]).assign(&x);
            y_train[[i]] = *y;
        } else {
            x_test.slice_mut(s![i - n_train, ..]).assign(&x);
            y_test[[i - n_train]] = *y;
        };
    }
    Dataset {
        x_train,
        y_train,
        x_test,
        y_test,
    }
}
