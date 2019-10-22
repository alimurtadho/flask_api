# Abstract

The project implements 4 linear models and one deep learning model using scikit-learn: Logistic Regression, Naïve Bayes, Support Vector Machine, K-Nearest Neighbours and Multi-Layer Perceptron Neural network to investigate their performance on diabetes and heart disease datasets obtained from the UCI data repository.

Multi-Layer Perceptron Neural Network outperforms other linear models however K-Nearest Neighbours gives identical results with less computing overhead. Performance improvements could also be achieved by using complex deep learning methods. 

# Experiment Results

## Cleveland Heart Disease Dataset

| Classifier    | Hyper-parameters |  Accuray (10-fold CV) |
| --------------------------| ---------------  | --------------------- |
| Logistic Regression | C = 0.13, Penalty = l1 | 83.84% |
| Linear SVC | C = 18.08 | 84.85% |
| Naïve Bayes | | 84.51% |
| K-Nearest Neighbors | n_neighbors=13, weights='uniform' | 85.52% |
| Multi-layer perceptron | learning_rate_init= 0.026958815931057856, hidden_layer_sizes = (29,26,5), learning_rate = constant, activation=identity, alpha = 16.681 | 86.2% |

## PIMA Indian Diabetes

| Classifier    | Hyper-parameters |  Accuray (10-fold CV) |
| --------------------------| ---------------  | --------------------- |
| Logistic Regression | C = 0.25999, Penalty = l2 | 77.6% |
| Linear SVC | C = 36.74999 | 78.77% |
| Naïve Bayes | | 75.651% |
| K-Nearest Neighbors | n_neighbors=27, weights='uniform' | 78.125% |
| Multi-layer perceptron | learning_rate_init= 0. 0.043967984410493759, hidden_layer_sizes = (23, 44), learning_rate = constant, activation=logistic, alpha = 0.1 | 79.03% |

# Deployment

1.  Install Python 3 and virtualenv

    a.	If the deployment is being done in ubuntu, run the following commands in the terminal
    ```bash
    sudo apt-get install python3 python3-pip python3-tk
    sudo pip install virtualenv
    ```
    b.	If the deployment is being done in windows, installing python is recommended using [Anaconda](https://www.continuum.io/downloads)
2.  Download the zip, or clone it using git.
    ```bash
    git clone https://github.com/nikhil-pandey/fyp-ml
    ```
3.  Create a virtual environment and install the dependencies.

    a.  In ubuntu, create the virtual environment for python 3 and activate it; then install the dependencies in requirements.txt file using the command
    ```bash
    pip install -r requirements.txt
    ```
    b.	For windows, refer to `requirements-anaconda.txt` file for creating virtual environment and installing dependencies.
4.  Run `app.py`.


# Live Demo
[Click Here](https://aimed.nikhil.com.np)

# Read Full Report
[Click Here](https://nikhil.com.np/storage/aimed.pdf)

# [License](LICENSE)
This project is open-sourced under the [MIT license](LICENSE)
