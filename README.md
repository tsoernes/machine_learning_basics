4# Machine learning basics

This repository contains implementations of basic machine learning algorithms in plain Rust. 
It is a fork of, and follows the spirit of, the original 
[machine learning basics in plain Python](). 
All algorithms are implemented from scratch without using additional machine learning libraries.
The intention is to provide a basic understanding of the algorithms and their underlying structure,
and how to port ML algorithms to Rust, *not* to provide the most efficient implementations. 

<!-- - [Linear Regression](linear_regression.ipynb) -->
<!-- - [Linear Regression](src/linear_regression.rs) -->
- [Logistic Regression](src/logistic_regression.rs) e.g. `cargo run -- lgr --n_iters 600 --learning_rate 0.009`
<!-- - [Perceptron](perceptron.ipynb) -->
- [K Nearest Neighbor](src/k_nearest_neighbors.rs) e.g. `cargo run -- knn -k 5`
- [K Means](src/k_means.rs) e.g. `cargo run -- kmc -k 4`
<!-- - [k-Means clustering](kmeans.ipynb) -->
<!-- - [Simple neural network with one hidden layer](simple_neural_net.ipynb) -->
<!-- - [Multinomial Logistic Regression](softmax_regression.ipynb) -->
<!-- - [Decision tree for classification](decision_tree_classification.ipynb) -->
- [Decision tree for regression](src/decision_tree_regression.rs) e.g. `cargo run -- dtr --max_depth 4 --min_samples 2`
  
## Contribute
Still missing:

- Linear Regression
- Perceptron
- k-Means clustering
- Simple neural network with one hidden layer
- Multinomial Logistic Regression
- Decision tree for classification
- Reinforcement learning (e.g. Q-learning with a linear neural network)
- Support Vector Machine
  
## Feedback

If you have a favorite algorithm that should be included or spot a mistake, please let me know by creating a new issue.

## License

See the LICENSE file for license rights and limitations (MIT).
