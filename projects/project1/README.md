# Machine Learning Project 1 (2023): 

Detailed project outlines and requirements can be found in the [project description](./project1_description.pdf). This challenge includes an [AIcrowd online contest] (https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

## Getting Started
Provided scripts and notebook files were built and tested with a conda environment with python version 3.9.16 
The following external libraries are used within the scripts:

```bash
numpy (as np)
matplotlib.pyplot (as plt)
```

## Running Prerequisites
Before running the scripts and notebook files, you should keep the folder structure under folder **scripts** as follows:

```bash
  .
  ├── optimal_weight.npy
  ├── loss.py
  ├── DataHandler.py
  ├── dataset
  │   ├── x_train.csv              # training set features, extract first
  │   ├── y_train.csv              # training set features, extract first
  │   └── x_test.csv               # testing set features, extract first
  ├── helpers.py
  ├── implementations.py
  ├── utils.py
  ├── plots.py
  ├── DataHandler.py
  └── run.py
```

All scripts are placed under the **scripts** folder, and you can find the code that generates our prediction file pred.csv in `run.py`.


## Implementation Details

#### [`run.py`](run.py)

The code for reproducing the best result. To run the script:

```bash
Python3 run.py
```

#### [`loss.py`](loss.py)

Script that contains the functions to calculate the loss functions for implemented machine learning algorithms.

#### [implementations.py](implementations.py)

Script that contains the implementation of machine learning algorithms according to the following table:

| Function            | Parameters | Details |
|-------------------- |-----------|---------|
| mean_squared_error_gd | `y, tx, initial_w, max_iters, gamma`  | Linear Regression by Gradient Descent |
| mean_squared_error_sgd | `y, tx, initial_w, max_iters, gamma`  | Linear Regression by Stochastic Gradient Descent |
| least_squares     | `y, tx` | Linear Regression by Solving Normal Equation |
| ridge_regression  | `y, tx, lambda_` | Ridge Regression by Soving Normal Equation |
| logistic_regression | `y, tx, initial_w, max_iters, gamma, threshold, batch_size`, verbose=False | Logistic Regression by Gradient Descent |
| reg_logistic_regression | `y, tx, lambda_, initial_w, max_iters, gamma, threshold, batch_size, verbose=False` | Regularised Logistic Regression by Gradient Descent |
| reg_focal_logistic_regression | `y, tx, initial_w, max_iters, gamma, lambda_, alpha, focal_gamma, verbose=False` | (Regularised) Logistic Regression by Gradient Descent with Focal Loss |

All functions returns a set of two key values: `w, loss`, where `w` indicates the last weight vector of the algorithm, and `loss` corresponds to this weight `w`.


#### [DataHandler.py](DataHandler.py)
 Class file that encapsulates all data processing steps.

#### [helpers.py](helpers.py)

Script that contains helper functions for loading the dataset and creating the prediction files (provided) .

#### [utils.py](utils.py)

Script that contains helper functions that are not originally provided.

#### [optimal_weight.npy](optimal_weight.npy)

File that contains optimal weight for the best model, for validation purpose.


## Best Performance
Our best model: Focal-loss Regularised Logistic Regression with One-hot Encoding and Mean Imputation, test F1 Score: 0.434, Accuracy: 0.872


## Authors
* [*Yixuan Xu*](https://github.com/Alvorecer721)
* [*Pingsheng Li*](https://github.com/Pingsheng-Kevin)
* Zhe Chen
