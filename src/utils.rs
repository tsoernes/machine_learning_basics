use super::RngSeed;
use ndarray::*;
use ndarray_rand::RandomExt;
use num_traits::identities::Zero;
use num_traits::Float;
use quickersort;
use rand::distributions::StandardNormal;
use rand::distributions::Uniform;
use rand::{thread_rng, ChaChaRng, Rng, SeedableRng};
use std::fmt::Debug;
use std::str::FromStr;

/// Computes the euclidean distance (aka L2 distance)
/// between two vectors
pub fn l2_distance(vec1: &ArrayView1<f64>, vec2: &ArrayView1<f64>) -> f64 {
    let mut x1: Array1<f64> = vec1 - vec2;
    x1.mapv_inplace(|e| e.powf(2.0));
    x1.scalar_sum().sqrt()
}

pub fn mean<D: Dimension>(arr: &Array<f64, D>) -> f64 {
    arr.scalar_sum() / arr.len() as f64
}

/// Argsort of a 1D array of floats
pub fn argsort_floats_1d<E: Float>(arr: &Array1<E>) -> Array1<usize> {
    let mut zipped: Array1<(usize, &E)> = arr.into_iter().enumerate().collect();

    quickersort::sort_by(
        &mut zipped.as_slice_mut().unwrap(),
        &|(_, x): &(_, &E), (_, y): &(_, &E)| match x.partial_cmp(y) {
            Some(ord) => ord,
            None => panic!("Attempting to sort NaN's"),
        },
    );
    zipped.map(|(i, _)| *i)
}

#[derive(Debug)]
pub struct Dataset<X, Y> {
    pub x_train: Array2<X>,
    pub y_train: Array1<Y>,
    pub x_test: Array2<X>,
    pub y_test: Array1<Y>,
}

/// Shuffle two arrays in unison
pub fn shuffle2<E1, E2, D1, D2>(
    arr1: Array<E1, D1>,
    arr2: Array<E2, D2>,
    rng_seed: Option<RngSeed>,
) -> (Array<E1, D1>, Array<E2, D2>)
where
    E1: Copy,
    E2: Copy,
    D1: Dimension + RemoveAxis,
    D2: Dimension + RemoveAxis,
{
    let mut indecies: Vec<usize> = (0..arr1.len_of(Axis(0))).collect();
    match rng_seed {
        Some(seed) => {
            ChaChaRng::from_seed(seed).shuffle(&mut indecies);
        }
        _ => {
            thread_rng().shuffle(&mut indecies);
        }
    };
    let arr1 = arr1.select(Axis(0), &indecies);
    let arr2 = arr2.select(Axis(0), &indecies);
    (arr1, arr2)
}

/// Shuffle the data set, then split into training and test set with ratio 'train_test_split_ratio'.
pub fn train_test_split<X, Y>(
    data_x: Array2<X>,
    data_y: Array1<Y>,
    train_test_split_ratio: f64,
) -> Dataset<X, Y>
where
    X: Copy + FromStr + Zero + Debug,
    Y: Copy + FromStr + Zero + Debug,
{
    let n_samples = data_x.rows();
    let n_features = data_x.cols();
    let n_train = (train_test_split_ratio * n_samples as f64) as usize;
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

///     Generate isotropic Gaussian blobs for clustering.
///     Parameters
///     ----------
///     n_samples : The total number of points equally divided among clusters.
///     n_features : The number of features for each sample.
///     centers : The number of centers to generate
///
///     Returns
///     -------
///     X : array of shape [n_samples, n_features]
///         The generated samples.
///     y : array of shape [n_samples]
///         The label of the samples.
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    n_centers: usize,
) -> (Array2<f64>, Array1<usize>) {
    let centers: Array2<f64> = Array::random((n_centers, n_features), Uniform::new(-10.0, 10.0));
    // let cluster_std: Array1<f64> = Array::ones(n_centers);
    let mut n_samples_per_center: Array1<usize> =
        Array::ones(n_centers) * (n_samples as f64 / n_centers as f64) as usize;
    for i in 0..(n_samples % n_centers) {
        n_samples_per_center[[i]] += 1;
    }
    let mut x = Array::zeros((n_samples, n_features));
    let mut y = Array::zeros(n_samples);
    let mut n_added = 0;
    for (i, n) in n_samples_per_center.into_iter().enumerate() {
        let noise: Array2<f64> = Array::random((*n, n_features), StandardNormal);
        let xi = noise + centers.slice(s![i, ..]);
        x.slice_mut(s![n_added..n_added + n, ..]).assign(&xi);
        y.slice_mut(s![n_added..n_added + n]).assign(&array![i]);
        n_added += n;
    }
    shuffle2(x, y, None)
}

/// The sigmoid function, also known as the logistic function
pub fn sigmoid(a: f64) -> f64 {
    1.0 / (1.0 + (-a).exp())
}
