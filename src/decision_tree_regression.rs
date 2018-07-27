use ndarray::*;
// Definition for binary tree:

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

struct Dataset<'a> {
    x_left: ArrayView1<'a, f64>,
    y_left: ArrayView1<'a, f64>,
    x_right: ArrayView1<'a, f64>,
    y_right: ArrayView1<'a, f64>,
}

fn mean<D: Dimension>(arr: &ArrayView<f64, D>) -> f64 {
    arr.scalar_sum() / arr.len() as f64
}

impl TreeNode {
    // fn new () -> Tree {
    //     Tree {
    //         value: v,
    //         left: None,
    //         right: None,
    //     }
    // }

    fn new(data: Dataset, depth: usize, max_depth: usize, min_samples: usize) -> TreeNode {
        // if depth >= max_depth
        TreeNode::Leaf {
            value: mean(&data.y_left),
        }
    }

    fn predict(&self, example: Array1<f64>) -> f64 {
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

// fn split<'a>(x: ArrayView1<'a, f64>, y: ArrayView1<'a, f64>) -> (usize, f64, f64, Dataset<'a>) {
//     let m = x.len() / 2;
//     let feature_idx = 0;
//     let threshhold = 0.0;
//     let cost = 0.0;
//     let dataset = Dataset {
//         x_left: x.slice(s![..m]),
//         y_left: y.slice(s![..m]),
//         x_right: x.slice(s![m..]),
//         y_right: y.slice(s![m..]),
//     };
//     (feature_idx, threshhold, cost, dataset)
// }

fn split<'a>(
    x: ArrayView1<'a, f64>,
    y: ArrayView1<'a, f64>,
) -> (
    ArrayView1<'a, f64>,
    ArrayView1<'a, f64>,
    ArrayView1<'a, f64>,
    ArrayView1<'a, f64>,
) {
    let m = x.len() / 2;
    // `x` does not live long enough (borrowed value does not live long enough) (rust-cargo)
    // `y` does not live long enough (borrowed value does not live long enough) (rust-cargo)
    return (
        x.slice(s![..m]),
        y.slice(s![..m]),
        x.slice(s![m..]),
        y.slice(s![m..]),
    );
}
