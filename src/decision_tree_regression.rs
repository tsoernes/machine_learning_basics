use super::RngSeed;
use csv;
use ndarray::*;
use ndarray_rand::RandomExt;
use rand::distributions::StandardNormal;
use std::f64;
use utils::{mean, shuffle2, train_test_split};

enum TreeNode {
    Leaf {
        value: f64,
    },
    Node {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

struct DataSplit {
    // x's have 1 sample per row, that is:
    // x[sample_idx][feature_idx] = feature_value
    x_left: Array2<f64>,
    // y[sample_idx] = target_value
    y_left: Array1<f64>,
    x_right: Array2<f64>,
    y_right: Array1<f64>,
}

impl TreeNode {
    /// Construct a new decision tree.
    /// 'x': features/inputs
    /// 'y': targets/outputs to regress
    /// 'max_depth': Maximum number of splits from root to leaf node that the tree can grow.
    ///    Lower values decrease overfitting.
    /// 'min_samples': The minimum number of samples left in the data set in
    ///    order to perform a split. Larger values decrease overfitting.
    pub fn new(x: Array2<f64>, y: Array1<f64>, max_depth: usize, min_samples: usize) -> TreeNode {
        assert!(max_depth >= 1);
        assert!(min_samples >= 1);
        TreeNode::_new(x, y, 1, max_depth, min_samples)
    }

    fn _new(
        x: Array2<f64>,
        y: Array1<f64>,
        depth: usize,
        max_depth: usize,
        min_samples: usize,
    ) -> TreeNode {
        let (feature_idx, threshold, dataset) = best_split(x, y);
        let (n_left_samples, n_right_samples) = (dataset.x_left.rows(), dataset.x_right.rows());
        let (left_node, right_node) = if depth >= max_depth {
            (
                TreeNode::new_terminal(&dataset.y_left),
                TreeNode::new_terminal(&dataset.y_right),
            )
        } else {
            // If there are enough samples remaining in the branch,
            // then construct the tree recursively.
            let left = if n_left_samples < min_samples {
                TreeNode::new_terminal(&dataset.y_left)
            } else {
                TreeNode::_new(
                    dataset.x_left,
                    dataset.y_left,
                    depth + 1,
                    max_depth,
                    min_samples,
                )
            };
            let right = if n_right_samples < min_samples {
                TreeNode::new_terminal(&dataset.y_right)
            } else {
                TreeNode::_new(
                    dataset.x_right,
                    dataset.y_right,
                    depth + 1,
                    max_depth,
                    min_samples,
                )
            };
            (left, right)
        };
        // Construct a new tree node. The left node classifies samples that have
        // features 'feature_idx' less than the threshold.
        TreeNode::Node {
            feature_idx,
            threshold,
            left: Box::new(left_node),
            right: Box::new(right_node),
        }
    }

    fn new_terminal(y: &Array1<f64>) -> TreeNode {
        TreeNode::Leaf { value: mean(y) }
    }

    /// Given a set of features 'example', predict the target value
    pub fn predict(&self, example: ArrayView1<f64>) -> f64 {
        // Recursively traverse the tree downwards until a leaf node is reached.
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Node {
                feature_idx,
                threshold,
                left,
                right,
            } => {
                if example[[*feature_idx]] < *threshold {
                    left.predict(example)
                } else {
                    right.predict(example)
                }
            }
        }
    }

    /// Evaluate decision tree performance on a data set
    pub fn test(&self, data_x: Array2<f64>, data_y: Array1<f64>) {
        let n_test = data_y.len();
        let mut mse = 0.0;
        for i in 0..n_test {
            let result = self.predict(data_x.slice(s![i, ..]));
            mse += (data_y[[i]] - result).powf(2.0);
        }
        mse *= 1.0 / n_test as f64;
        println!("{:?}", mse);
    }
}

/// Split the data set into two; the left set containing the entries with the given feature
/// valued less than the threshold, and the right set the entries greater than
/// the threshold.
fn split(x: &Array2<f64>, y: &Array1<f64>, feature_idx: usize, threshold: f64) -> DataSplit {
    let (mut lt, mut gt): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    for (i, row) in x.outer_iter().enumerate() {
        if row[[feature_idx]] < threshold {
            lt.push(i);
        } else {
            gt.push(i);
        }
    }
    DataSplit {
        x_left: x.select(Axis(0), &lt),
        y_left: y.select(Axis(0), &lt),
        x_right: x.select(Axis(0), &gt),
        y_right: y.select(Axis(0), &gt),
    }
}

/// Find the best feature and feature threshold to split on.
fn best_split(x: Array2<f64>, y: Array1<f64>) -> (usize, f64, DataSplit) {
    let mut best_feature_idx = 0;
    let mut best_threshold = x[[0, 0]];
    let mut best_dataset = split(&x, &y, best_feature_idx, best_threshold);
    let mut best_cost = f64::MAX;
    let rs = x.rows();
    for feature_idx in 0..x.cols() {
        for sample_idx in 0..rs {
            let threshold = x[[sample_idx, feature_idx]];
            let dataset = split(&x, &y, feature_idx, threshold);
            let cost = get_cost(&dataset.y_left, &dataset.y_right);
            if cost < best_cost {
                best_feature_idx = feature_idx;
                best_threshold = threshold;
                best_dataset = dataset;
                best_cost = cost;
            }
        }
    }
    (best_feature_idx, best_threshold, best_dataset)
}

/// The Mean Squared Error for a given split, pretending that each node after the split
/// is a terminal node. The MSE for each subbranch is
/// normalized by how many samples end up in the branch and then added together.
fn get_cost(y_left: &Array1<f64>, y_right: &Array1<f64>) -> f64 {
    // The MSE on the given targets (which are from the training data set),
    // assuming the node is a terminal node
    fn mse(y: &Array1<f64>, n: usize) -> f64 {
        let inv = 1.0 / n as f64;
        let y_hat = inv * y.scalar_sum();
        inv * (y - y_hat).mapv(|e| e.powf(2.0)).scalar_sum()
    }
    let (n_left, n_right) = (y_left.len(), y_right.len());
    let mse_left = if n_left > 0 { mse(y_left, n_left) } else { 0.0 };
    let mse_right = if n_right > 0 {
        mse(y_right, n_right)
    } else {
        0.0
    };
    let (n_left, n_right) = (n_left as f64, n_right as f64);
    let n_total = n_left + n_right;
    (n_left / n_total) * mse_left + (n_right / n_total) * mse_right
}

/// Load the Boston Housing data set from file,
/// build the decision tree with the given parameters
/// and test how the decision tree performs.
/// TODO load boston into python original; compare results
pub fn run(
    max_depth: usize,
    min_samples: usize,
    train_test_split_ratio: f64,
    rng_seed: Option<RngSeed>,
) {
    let file_path = "datasets/boston.csv";
    let mut rdr = csv::Reader::from_path(file_path).unwrap();
    let n_samples = 333; // rdr.records().count();
    let n_features = rdr.headers().unwrap().len() - 1;
    let mut data_x: Array2<f64> = Array::zeros((n_samples, n_features + 1));
    let mut data_y: Array1<f64> = Array::zeros(n_samples);
    for (i, result) in rdr.records().enumerate() {
        let r = result.unwrap();
        // println!("{}, {:?}", i, r);
        let y: f64 = r.get(n_features).expect("idx").parse().expect("parse");
        data_y[[i]] = y;
        let x: Array1<f64> = r.into_iter()
            .take(n_features)
            .map(|e| e.parse().unwrap())
            .collect();
        data_x.slice_mut(s![i, ..-1]).assign(&x);
    }
    let (data_x, data_y) = shuffle2(data_x, data_y, rng_seed);
    let dataset = train_test_split(data_x, data_y, train_test_split_ratio);
    let dtree = TreeNode::new(dataset.x_train, dataset.y_train, max_depth, min_samples);
    dtree.test(dataset.x_test, dataset.y_test);
}

/// Same benchmark as in original repo
pub fn run_rand(
    max_depth: usize,
    min_samples: usize,
    train_test_split_ratio: f64,
    _rng_seed: Option<RngSeed>,
) {
    let x: Array1<f64> = Array::linspace(3.0, 3.0, 400);
    let y = x.mapv(|e| e.powf(2.0)) + Array::random(400, StandardNormal);
    let x = x.into_shape((400, 1)).unwrap();
    let dataset = train_test_split(x, y, train_test_split_ratio);
    let dtree = TreeNode::new(dataset.x_train, dataset.y_train, max_depth, min_samples);
    dtree.test(dataset.x_test, dataset.y_test);
}
