# Multi-Layer Perceptron (MLP) Implementation for Classification

This project implements a Multi-Layer Perceptron (MLP) from scratch for classification tasks. It includes data loading, preprocessing, forward and backward propagation, weight updates, and hyperparameter tuning using RandomizedSearchCV.  Additionally, it compares the performance of the best and worst models found during hyperparameter optimization.

## Project Description

This project was developed as part of a Knowledge Discovery and Data Mining course. The goal is to build an MLP model to classify data, evaluate its performance, and understand the impact of hyperparameter selection. The implementation covers the essential components of a neural network, including:

* **Data Loading and Preprocessing:** Loading data from Excel files and preprocessing it by normalizing features.
* **MLP Architecture:** Defining the structure of the MLP with input, hidden, and output layers.
* **Activation Functions:** Implementing sigmoid and softmax activation functions.
* **Forward Propagation:** Calculating the output of the network given an input.
* **Backward Propagation:** Computing the gradients of the loss function with respect to the network's parameters.
* **Weight Updates:** Updating the network's parameters to minimize the loss.
* **Hyperparameter Tuning:** Using RandomizedSearchCV to find the optimal hyperparameters.
* **Performance Evaluation:** Evaluating the model's performance using loss curves.
* **Comparison of Best and Worst Models:** Analyzing the training and validation loss of the best and worst performing models from hyperparameter tuning.

## Files

* `MLPRandomizedSearchCV.ipynb`:  Jupyter Notebook containing the Python implementation of the MLP.
* `DataSets/THA2train.xlsx`: Excel file containing the training dataset. *(Please provide this file if you want the code to run)*
* `DataSets/THA2validate.xlsx`: Excel file containing the validation dataset. *(Please provide this file if you want the code to run)*
* `README.md`:  This file, providing an overview of the project.

## Dependencies

* `numpy`:  For numerical computations.
* `matplotlib`:  For plotting.
* `scikit-learn (sklearn)`: For model selection and datasets.
* `pandas`: For data manipulation (specifically reading excel files)

## Usage

1.  **Install Dependencies:**
    ```bash
    pip install numpy matplotlib scikit-learn pandas
    ```
2.  **Place Data:**
    Ensure that `THA2train.xlsx` and `THA2validate.xlsx` are located in the `DataSets/` directory relative to the notebook.
3.  **Run the Notebook:**
    Open and execute `MLPRandomizedSearchCV.ipynb` in a Jupyter environment. The notebook will:
    * Load and preprocess the data.
    * Define and train the MLP model.
    * Tune hyperparameters using RandomizedSearchCV.
    * Evaluate performance and generate plots.

## Code Explanation

The Jupyter Notebook (`MLPRandomizedSearchCV.ipynb`) is structured as follows:

1.  **Import Libraries:** Imports necessary libraries like NumPy, Matplotlib, and scikit-learn.
2.  **Data Loading:** Reads the training and validation data from Excel files using pandas.  *(Note:  The pandas import statement is missing in the provided notebook, you might need to add `import pandas as pd`)*
3.  **Data Preprocessing:**
    * Separates features (X) and labels (y) for both training and validation sets.
    * Converts categorical labels into a one-hot encoded format using `pd.get_dummies`.
    * Normalizes the features by subtracting the mean and dividing by the standard deviation.
4.  **Activation Functions:**
    * Defines the `sigmoid` function, with overflow handling, to introduce non-linearity in the hidden layer.
    * Defines `sigmoid_derivative` to compute the derivative of the sigmoid function, used in backpropagation.
    * Defines the `softmax` function to produce probability distributions for the output layer in multi-class classification.
5.  **MLP Class:**
    * The `MLP` class encapsulates the neural network logic.
    * `__init__`:  Initializes the weights and biases with random values. Weights are initialized using a method that takes into account the size of the layers to improve training.
    * `forward`:  Performs forward propagation, calculating the output of each layer.
    * `backward`:  Implements backpropagation to compute the gradients of the loss with respect to the weights and biases.
    * `update_weights`:  Updates the weights and biases using the calculated gradients and the learning rate.
    * `compute_loss`:  Calculates the cross-entropy loss, a standard loss function for classification problems.
6.  **Hyperparameter Tuning:**
    * Defines a parameter distribution (`param_dist`) for RandomizedSearchCV to explore different combinations of `hidden_size` and `learning_rate`.
    * Uses `RandomizedSearchCV` to find the best hyperparameters based on cross-validation accuracy.
    * Trains the best and worst models found during the search.
7.  **Training Loop:**
    * Iterates over a specified number of epochs.
    * Implements mini-batch gradient descent to train the models.
    * Calculates and stores training and validation losses.
    * Prints training progress periodically.
8.  **Results and Visualization:**
    * Plots the training and validation loss curves for both the best and worst models to visualize the learning process and compare performance.
    * Potentially includes additional evaluation metrics (commented out in the provided notebook).

## Notes

* Error Handling: The notebook throws a `NameError` because pandas is used without being imported. Add `import pandas as pd` at the beginning of the notebook.
* Data Availability:  The code requires the `THA2train.xlsx` and `THA2validate.xlsx` files to run.  Please provide these files.
* Potential Improvements: The code could be extended to include:
    * More sophisticated hyperparameter tuning techniques.
    * Regularization methods to prevent overfitting.
    * Different optimization algorithms.
    * Additional performance metrics (e.g., precision, recall, F1-score).
