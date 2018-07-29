use super::RNG_SEED;
use csv;
use ndarray::*;
use rand::{thread_rng, ChaChaRng, Rng, SeedableRng};
use std::f64;

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

fn mean<D: Dimension>(arr: &Array<f64, D>) -> f64 {
    arr.scalar_sum() / arr.len() as f64
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
}

/// Split the data set into two; the left set containing the entries with features
/// less than the threshold, and the right set the entries with features greater than
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
    let xl = x.select(Axis(0), &lt);
    let yl = y.select(Axis(0), &lt);
    let xr = x.select(Axis(0), &gt);
    let yr = y.select(Axis(0), &gt);
    DataSplit {
        x_left: xl,
        y_left: yl,
        x_right: xr,
        y_right: yr,
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

/// The Mean Squared Error for a given split. The MSE for each subbranch is
/// normalized by how many samples end up in the branch and then added together.
fn get_cost(y_left: &Array1<f64>, y_right: &Array1<f64>) -> f64 {
    // The MSE on the given targets (which are from the training data set),
    // provided the node is a terminal node
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
pub fn run(max_depth: usize, min_samples: usize) {
    let file_path = "datasets/boston.csv";
    let n_samples: usize = 333; // Data set above has 333 entries
    let n_features = 14;
    // Use 90 % of the data set for training and 10 % for testing
    let train_test_split = 0.9;
    let mut rdr = csv::Reader::from_path(file_path).unwrap();

    let mut data: Array2<f64> = Array::zeros((n_samples, n_features + 1));
    for (i, result) in rdr.records().enumerate() {
        let row: Array1<f64> = result
            .unwrap()
            .into_iter()
            .map(|e| e.parse().unwrap())
            .collect();
        data.slice_mut(s![i, ..]).assign(&row);
    }

    // Shuffle the data set
    let mut indecies: Vec<usize> = (0..n_samples).collect();
    let mut rng = ChaChaRng::from_seed(RNG_SEED);
    // let mut rng = thread_rng();
    rng.shuffle(&mut indecies);
    let data = data.select(Axis(0), &indecies);

    // Split data set into test and training set.
    // Split sets into features (input) x and targets y
    let n_train = (train_test_split * n_samples as f64) as usize;
    let n_test = n_samples - n_train;
    let mut x_train: Array2<f64> = Array::zeros((n_train, n_features));
    let mut y_train: Array1<f64> = Array::zeros(n_train);
    let mut x_test: Array2<f64> = Array::zeros((n_test, n_features));
    let mut y_test: Array1<f64> = Array::zeros(n_test);
    for (i, row) in data.outer_iter().enumerate() {
        if i < n_train {
            x_train.slice_mut(s![i, ..]).assign(&row.slice(s!(..-1)));
            y_train.slice_mut(s![i]).assign(&row.slice(s!(-1)));
        } else {
            x_test
                .slice_mut(s![i - n_train, ..])
                .assign(&row.slice(s!(..-1)));
            y_test.slice_mut(s![i - n_train]).assign(&row.slice(s!(-1)));
        };
    }

    let dtree = TreeNode::new(x_train, y_train, max_depth, min_samples);
    // Evaluate decision tree performance; in the case of regression we
    // often use average mean squared error
    let mut mse = 0.0;
    for i in 0..n_test {
        let result = dtree.predict(x_test.slice(s![i, ..]));
        mse += (y_test[[i]] - result).powf(2.0);
    }
    mse *= 1.0 / n_test as f64;
    println!("{:?}", mse);
}
