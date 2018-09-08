use super::RngSeed;
use ndarray::*;
use std::f64;
use utils::{l2_distance, make_blobs, shuffle2, train_test_split};

/// K Means unsupervised clustering
struct KMeans {
    centers: Array2<f64>,
    k: usize,
}

impl KMeans {
    pub fn new(x: Array2<f64>, k: usize) -> KMeans {
        // Initialize centers by picking random samples from the data set
        let centers_i: Vec<usize> = (0..k).collect();
        let mut centers: Array2<f64> = x.select(Axis(0), &centers_i);

        // Keep track of whether the assignment of data points
        // to the clusters has changed. When it stops changing, the model has converged.
        let mut assigns: Array1<usize> = Array::zeros(x.rows());

        loop {
            // Assign each datapoint to the closest cluster.
            let new_assigns: Array1<usize> = x.outer_iter()
                .map(|example| KMeans::_predict(&centers, example))
                .collect();
            if assigns == new_assigns {
                break;
            }
            // Update the current estimates of the cluster centers by setting
            // them to the mean of all instance belonging to that cluster
            for id in 0..k {
                // Indecies of data points classified as 'id'
                let points_idx: Vec<usize> = new_assigns
                    .iter()
                    .enumerate()
                    .filter(|(_, class)| **class == id)
                    .map(|(i, _)| i)
                    .collect();
                assert!(
                    points_idx.len() > 0,
                    format!("Nothing classified as class: {}", id)
                );
                let datapoints = x.select(Axis(0), &points_idx);
                assert!(datapoints.len() > 0);
                // The new center is the mean of the data points classified as 'id'
                let n = datapoints.len_of(Axis(0)) as f64;
                let new_center = datapoints.sum_axis(Axis(0)) / n;
                centers.slice_mut(s![id, ..]).assign(&new_center);
            }
            assigns = new_assigns;
        }
        KMeans { centers, k }
    }

    fn _predict(centers: &Array2<f64>, example: ArrayView1<f64>) -> usize {
        // Compute the L2 distance from the example to every cluster.
        // The nearest cluster is our prediction
        centers
            .outer_iter()
            .map(|cluster| l2_distance(&cluster, &example))
            .enumerate()
        // Argmin over iterator (index of cluster with shortest distance from example)
            .fold(
                (0, f64::MAX),
                |(imin, emin), (i, e)| if e < emin { (i, e) } else { (imin, emin) },
            )
            .0
    }
    /// Given a set of features 'example', output a classification
    pub fn predict(&self, example: ArrayView1<f64>) -> usize {
        KMeans::_predict(&self.centers, example)
    }

    pub fn test(&self, data_x: Array2<f64>, data_y: Array1<usize>) {
        let preds: Array1<usize> = data_x
            .outer_iter()
            .map(|example| self.predict(example))
            .collect();
        // We don't know the correspondence between predicted and actual labels.
        // Finding the best correspondence requires testing (k!) combinations.
        // We use "Heap's algorithm" for creating all possible permutations
        // of the range 0..k
        let mut mapping: Array1<usize> = Array::from_vec((0..self.k).collect());
        let mut c = Array::zeros(self.k);
        let mut i = 0;
        let mut n_correct_best = 0;
        while i < self.k {
            if c[[i]] < i {
                // Find the amount of correct predictions for the current mapping
                let n_correct = preds
                    .iter()
                    .zip(data_y.iter())
                    .fold(0, |n, (pred, actual)| {
                        if mapping[[*pred]] == *actual {
                            n + 1
                        } else {
                            n
                        }
                    });
                if n_correct > n_correct_best {
                    n_correct_best = n_correct;
                }
                if i % 2 == 0 {
                    mapping.swap(0, i);
                } else {
                    mapping.swap(c[[i]], i);
                }
                c[[i]] += 1;
                i = 0;
            } else {
                c[[i]] = 0;
                i += 1;
            }
        }
        let acc = n_correct_best as f64 / data_y.len() as f64;
        println!(
            "{} of {} correct, accuracy: {} %",
            n_correct_best,
            data_y.len(),
            acc * 100.0
        );
    }
}

pub fn run(k: usize, train_test_split_ratio: f64, rng_seed: Option<RngSeed>) {
    let (x, y): (Array2<f64>, Array1<usize>) = make_blobs(1000, 2, k);
    let (x, y) = shuffle2(x, y, rng_seed);
    let dataset = train_test_split(x, y, train_test_split_ratio);
    let kmeans = KMeans::new(dataset.x_train, k);
    kmeans.test(dataset.x_test, dataset.y_test);
}
