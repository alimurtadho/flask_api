import os
import pickle
from sklearn.externals import joblib

config = {
    'heart': {
        'scalar_file': 'production/heart/standard_scalar.pkl',
        'LinearSVC': 'production/heart/LinearSVC.pkl',
        'LogisticRegression': 'production/heart/LogisticRegression.pkl',
        'NaiveBayes': 'production/heart/NaiveBayes.pkl',
        'KNeighbors': 'production/heart/KNeighbors.pkl',
        'NeuralNetwork': 'production/heart/NN.pkl',
        'Ensemble' : 'production/heart/Ensemble.pkl'
    },
    'diabetes': {
        'scalar_file': 'production/diabetes/standard_scalar.pkl',
        'LinearSVC': 'production/diabetes/LinearSVC.pkl',
        'LogisticRegression': 'production/diabetes/LogisticRegression.pkl',
        'NaiveBayes': 'production/diabetes/NaiveBayes.pkl',
        'KNeighbors': 'production/diabetes/KNeighbors.pkl',
        'NeuralNetwork': 'production/diabetes/NN.pkl',
        'Ensemble' : 'production/diabetes/Ensemble.pkl'
    }
}

dir = os.path.dirname(__file__)

def GetJobLibFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return joblib.load(os.path.join(dir, filepath))
    return None

def GetPickleFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return pickle.load( open(os.path.join(dir, filepath), "rb" ) )
    return None

def GetStandardScalarForHeart():
    return GetPickleFile(config['heart']['scalar_file'])

def GetAllClassifiersForHeart():
    return (GetLinearSVCClassifierForHeart(), GetLogisticRegressionClassifierForHeart(), GetNaiveBayesClassifierForHeart(), GetKNeighborsClassifierForHeart(), GetNeuralNetworkClassifierForHeart(), GetEnsembleClassifierForHeart())

def GetLinearSVCClassifierForHeart():
    return GetJobLibFile(config['heart']['LinearSVC'])

def GetLogisticRegressionClassifierForHeart():
    return GetJobLibFile(config['heart']['LogisticRegression'])

def GetNaiveBayesClassifierForHeart():
    return GetJobLibFile(config['heart']['NaiveBayes'])

def GetKNeighborsClassifierForHeart():
    return GetJobLibFile(config['heart']['KNeighbors'])

def GetNeuralNetworkClassifierForHeart():
    return GetJobLibFile(config['heart']['NeuralNetwork'])

def GetEnsembleClassifierForHeart():
    return GetJobLibFile(config['heart']['Ensemble'])

## Diabetes

def GetAllClassifiersForDiabetes():
    return (GetLinearSVCClassifierForDiabetes(), GetLogisticRegressionClassifierForDiabetes(), GetNaiveBayesClassifierForDiabetes(), GetKNeighborsClassifierForDiabetes(), GetNeuralNetworkClassifierForDiabetes(), GetEnsembleClassifierForDiabetes())

def GetStandardScalarForDiabetes():
    return GetPickleFile(config['diabetes']['scalar_file'])

def GetLinearSVCClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['LinearSVC'])

def GetLogisticRegressionClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['LogisticRegression'])

def GetNaiveBayesClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['NaiveBayes'])

def GetKNeighborsClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['KNeighbors'])

def GetNeuralNetworkClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['NeuralNetwork'])

def GetEnsembleClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['Ensemble'])
