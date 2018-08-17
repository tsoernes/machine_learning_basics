use ndarray::*;
use std::ops::SubAssign;
use utils::{make_blobs, shuffle2, train_test_split};

/// The sigmoid function, also known as the logistic function
fn sigmoid(a: f64) -> f64 {
    1.0 / (1.0 + (-a).exp())
}

struct LogisticRegressor {
    weights: Array1<f64>,
    bias: f64,
}

impl LogisticRegressor {
    /// Construct and train a logistic regressor.
    pub fn new(
        x: Array2<f64>,
        y: Array1<f64>,
        n_iters: usize,
        learning_rate: f64,
    ) -> LogisticRegressor {
        let scale = 1.0 / x.rows() as f64;
        let n_features = x.cols();
        let mut lgr = LogisticRegressor {
            weights: Array::zeros(n_features),
            bias: 0.0,
        };
        for i in 0..n_iters {
            let y_pred = lgr.act(&x);
            // Negative categorical cross entropy for each data point
            let crents = y.clone() * y_pred.mapv(f64::ln)
                + (1.0 - y.clone()) * (1.0 - y_pred.clone()).mapv(f64::ln);
            // Average cross entropy for data set
            let cost = -scale * crents.scalar_sum();
            // Compute gradients for weights and bias
            let diff = y_pred - y.clone();
            let dw: Array1<f64> = scale * diff.dot(&x);
            let db: f64 = scale * diff.scalar_sum();
            // Update parameters with (non-stochastic) gradient descent
            lgr.weights.sub_assign(&(learning_rate * dw));
            lgr.bias.sub_assign(learning_rate * db);
            if i % 100 == 0 {
                println!("Cost iteration {}: {}", i, cost);
            }
        }
        lgr
    }

    /// Given a matrix [n_samples, n_features] of examples 'x',
    /// for each examples (row) compute the sigmoid of a linear combination
    /// of the example. Returns a matrix of size [n_samples]
    fn act<S: Data<Elem = f64>>(&self, x: &ArrayBase<S, Ix2>) -> Array1<f64> {
        let mut out = x.dot(&self.weights) + self.bias;
        out.mapv_inplace(sigmoid);
        out
    }

    /// Predicts binary label for an example.
    pub fn predict<S: Data<Elem = f64>>(&self, example: ArrayBase<S, Ix1>) -> f64 {
        let mut y_pred = self.act(&example.insert_axis(Axis(0)));
        // Threshold to 1's and 0's
        y_pred.mapv_inplace(|e| if e > 0.5 { 1.0 } else { 0.0 });
        y_pred[[0]]
    }

    /// Evaluate regressor performance on a data set
    pub fn test(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let y_preds: Array1<f64> = x.outer_iter()
            .map(|example| self.predict(example))
            .collect();
        let acc = 100.0 - (y - &y_preds).mapv(f64::abs).mean_axis(Axis(0)) * 100.0;
        acc[[]]
    }
}

pub fn run(
    n_iters: usize,
    learning_rate: f64,
    train_test_split_ratio: f64,
    rng_seed: Option<[u8; 32]>,
) {
    let (x, y): (Array2<f64>, Array1<usize>) = make_blobs(1000, 2, 2);
    let (x, y) = shuffle2(x, y, rng_seed);
    let y = y.mapv(|e| e as f64);
    let dataset = train_test_split(x, y, train_test_split_ratio);
    let lgr = LogisticRegressor::new(
        dataset.x_train.clone(),
        dataset.y_train.clone(),
        n_iters,
        learning_rate,
    );
    println!(
        "Training set accuracy: {} %",
        lgr.test(&dataset.x_train, &dataset.y_train)
    );
    println!(
        "Test set accuracy: {} %",
        lgr.test(&dataset.x_test, &dataset.y_test)
    );;
}
