# Machine learning basics

This repository contains implementations of basic machine learning algorithms in plain Rust. 
It is a fork of, and follows the spirit of, the original 
[machine learning basics in plain Python](). 
All algorithms are implemented from scratch without using additional machine learning libraries.
The intention is to provide a basic understanding of the algorithms and their underlying structure,
and how to port ML algorithms to Rust, *not* to provide the most efficient implementations. 

<!-- - [Linear Regression](linear_regression.ipynb) -->
<!-- - [Linear Regression](src/linear_regression.rs) -->
<!-- - [Logistic Regression](logistic_regression.ipynb) -->
<!-- - [Perceptron](perceptron.ipynb) -->
- [K Nearest Neighbor](src/k_nearest_neighbour.rs) e.g. `cargo run -- knn -k 5`
<!-- - [k-Means clustering](kmeans.ipynb) -->
<!-- - [Simple neural network with one hidden layer](simple_neural_net.ipynb) -->
<!-- - [Multinomial Logistic Regression](softmax_regression.ipynb) -->
<!-- - [Decision tree for classification](decision_tree_classification.ipynb) -->
- [Decision tree for regression](src/decision_tree_regression.rs) e.g. `cargo run -- dtr --max_depth 4 --min_samples 2`
  
  
## Feedback

If you have a favorite algorithm that should be included or spot a mistake, please let me know by creating a new issue.

## License

See the LICENSE file for license rights and limitations (MIT).
