use csv;
use ndarray::*;
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
    // x[sample_idx][feature_idx]
    x_left: Array2<f64>,
    // y[sample_idx]
    y_left: Array1<f64>,
    x_right: Array2<f64>,
    y_right: Array1<f64>,
}

fn mean<D: Dimension>(arr: &Array<f64, D>) -> f64 {
    arr.scalar_sum() / arr.len() as f64
}

impl TreeNode {
    pub fn new(
        x: Array2<f64>,
        y: Array1<f64>,
        depth: usize,
        max_depth: usize,
        min_samples: usize,
    ) -> TreeNode {
        let (feature_idx, threshold, _, dataset) = best_split(x, y);
        let (n_left_samples, n_right_samples) = (dataset.x_left.rows(), dataset.x_right.rows());
        let (left, right) = if (n_left_samples, n_right_samples) == (0, 0) {
            panic!("This case could happen, after all.");
        } else if depth >= max_depth {
            (
                TreeNode::new_terminal(&dataset.y_left),
                TreeNode::new_terminal(&dataset.y_right),
            )
        } else {
            let left = if n_left_samples < min_samples {
                TreeNode::new_terminal(&dataset.y_left)
            } else {
                TreeNode::new(
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
                TreeNode::new(
                    dataset.x_right,
                    dataset.y_right,
                    depth + 1,
                    max_depth,
                    min_samples,
                )
            };
            (left, right)
        };
        TreeNode::Node {
            feature_idx,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn new_terminal(y: &Array1<f64>) -> TreeNode {
        TreeNode::Leaf { value: mean(y) }
    }

    pub fn predict(&self, example: ArrayView1<f64>) -> f64 {
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

fn best_split(x: Array2<f64>, y: Array1<f64>) -> (usize, f64, f64, DataSplit) {
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
    (best_feature_idx, best_threshold, best_cost, best_dataset)
}

fn mean_squared_error(y_left: &Array1<f64>, y_right: &Array1<f64>) -> (f64, f64) {
    let (n_left, n_right) = (y_left.len(), y_right.len());

    fn mse(y: &Array1<f64>, n: usize) -> f64 {
        let inv = 1.0 / n as f64;
        let y_hat = inv * y.scalar_sum();
        inv * (y - y_hat).mapv(|e| e.powf(2.0)).scalar_sum()
    }
    let mse_left = if n_left > 0 { mse(y_left, n_left) } else { 0.0 };
    let mse_right = if n_right > 0 {
        mse(y_right, n_right)
    } else {
        0.0
    };
    (mse_left, mse_right)
}

fn get_cost(y_left: &Array1<f64>, y_right: &Array1<f64>) -> f64 {
    let (n_left, n_right) = (y_left.len() as f64, y_right.len() as f64);
    let n_total = n_left + n_right;
    let (mse_left, mse_right) = mean_squared_error(y_left, y_right);
    (n_left / n_total) * mse_left + (n_right / n_total) * mse_right
}

pub fn run(max_depth: usize, min_samples: usize) {
    let file_path = "datasets/boston.csv";
    let mut rdr = csv::Reader::from_path(file_path).unwrap();

    let n_train = (0.9 * 333.0) as usize;
    let n_test = 333 - n_train;
    // TODO these should probably be shuffled
    let mut x_train: Array2<f64> = Array::zeros((n_train, 14));
    let mut y_train: Array1<f64> = Array::zeros(n_train);
    let mut x_test: Array2<f64> = Array::zeros((n_test, 14));
    let mut y_test: Array1<f64> = Array::zeros(n_test);
    for (i, result) in rdr.records().enumerate() {
        let row: Array1<f64> = result
            .unwrap()
            .into_iter()
            .map(|e| e.parse().unwrap())
            .collect();
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

    let dtree = TreeNode::new(x_train, y_train, 0, max_depth, min_samples);
    let mut mse = 0.0;
    for i in 0..n_test {
        let result = dtree.predict(x_test.slice(s![i, ..]));
        mse += (y_test[[i]] - result).powf(2.0);
    }
    mse *= 1.0 / n_test as f64;
    println!("{:?}", mse);
}
