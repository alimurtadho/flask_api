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

