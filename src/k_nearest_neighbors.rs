use csv;
use ndarray::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::AddAssign;
use utils::{argsort_floats_1d, shuffle_split};

/// Computes the euclidean distance between each example in the training data
/// and a new input example
fn euclid_distance(data_x: &Array2<f64>, example: &ArrayView1<f64>) -> Array1<f64> {
    let mut x1: Array2<f64> = data_x - example;
    x1.mapv_inplace(|e| e.powf(2.0));
    let mut x3: Array1<f64> = x1.map_axis(Axis(0), |row| row.scalar_sum());
    x3.mapv_inplace(|e| e.sqrt());
    x3
}

/// Given the data set 'data_x' and 'data_y', predict the label of 'example'
/// using the 'k' nearest neighbors
fn predict<Y: Clone + Hash + Eq>(
    data_x: &Array2<f64>,
    data_y: &Array1<Y>,
    example: &ArrayView1<f64>,
    k: usize,
) -> Y {
    let dists = euclid_distance(data_x, example);
    let mut dists_ix = argsort_floats_1d(&dists);
    // Indecies of the 'k' smallest distance data entries
    dists_ix.slice_inplace(s![..k]);
    // Labels of 'k' nearest entries
    let k_ys: Array1<Y> = dists_ix.map(|ix| data_y[[*ix]].clone());
    // Group targets by their count
    let mut y_counts: HashMap<Y, usize> = HashMap::new();
    for y in k_ys.iter() {
        if y_counts.contains_key(&y) {
            y_counts.get_mut(&y).unwrap().add_assign(1);
        } else {
            // TODO should this not be possible without cloning, since its into_iter
            y_counts.insert(y.clone(), 1);
        }
    }
    // Return the most common label among 'k' nearest neighbors
    y_counts
        .into_iter()
        .max_by(|(_, cnt1), (_, cnt2)| cnt1.cmp(&cnt2))
        .unwrap()
        .0
}

pub fn run(k: usize, train_test_split: f64, rng_seed: Option<[u8; 32]>) {
    // Load data from csv file into arrays
    let file_path = "datasets/digits.csv";
    let mut rdr = csv::Reader::from_path(file_path).unwrap();
    let n_samples = 1796;
    // let n_samples = rdr.records().count();
    let n_features = rdr.headers().unwrap().len() - 1;
    let mut data_x: Array2<f64> = Array::zeros((n_samples, n_features + 1));
    let mut data_y: Array1<u32> = Array::zeros(n_samples);
    for (i, result) in rdr.records().enumerate() {
        let r = result.unwrap();
        // The last entry of each row in the csv file is the target/label
        let y = r.get(n_features).expect("idx").parse().expect("parse");
        data_y[[i]] = y;
        let x: Array1<f64> = r.into_iter()
            .take(n_features)
            .map(|e| e.parse().unwrap())
            .collect();
        data_x.slice_mut(s![i, ..-1]).assign(&x);
    }
    let dataset = shuffle_split(data_x, data_y, train_test_split, rng_seed);

    // Compute accuracy on test set
    let y_preds: Array1<u32> = dataset
        .x_test
        .outer_iter()
        .map(|example| predict(&dataset.x_train, &dataset.y_train, &example, k))
        .collect();
    let n_eq = y_preds.into_iter().zip(dataset.y_test.into_iter()).fold(
        0,
        |acc, (y_pred, y_targ)| if y_pred == y_targ { acc + 1 } else { acc },
    );
    let test_acc: f64 = (n_eq as f64) / (y_preds.len() as f64) * 100.00;
    println!("{}", test_acc);
}
