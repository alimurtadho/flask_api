import os
import pickle
from sklearn.externals import joblib

config = {
    'heart': {
        'scalar_file': 'production/heart/standard_scalar.pkl',
        'LinearSVC': 'production/heart/LinearSVC.pkl',
        'LogisticRegression': 'production/heart/LogisticRegression.pkl',
        'RandomForest': 'production/heart/RandomForest.pkl',
        'KNeighbors': 'production/heart/KNeighbors.pkl',
        'GradientBoasting': 'production/heart/NN.pkl',
        'Ensemble' : 'production/heart/Ensemble.pkl'
    },
    'diabetes': {
        'scalar_file': 'production/diabetes/standard_scalar.pkl',
        'LinearSVC': 'production/diabetes/LinearSVC.pkl',
        'LogisticRegression': 'production/diabetes/LogisticRegression.pkl',
        'RandomForest': 'production/diabetes/RandomForest.pkl',
        'KNeighbors': 'production/diabetes/KNeighbors.pkl',
        'GradientBoasting': 'production/diabetes/GradientBoasting.pkl',
        'Ensemble' : 'production/diabetes/Ensemble.pkl',
        # 'Gradient' : 'production/diabetes/gradientboasting.pkl'
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
    return (GetLinearSVCClassifierForHeart(), GetLogisticRegressionClassifierForHeart(), GetRandomForestClassifierForHeart(), GetKNeighborsClassifierForHeart(), GetGradientBoastingClassifierForHeart(), GetEnsembleClassifierForHeart())

def GetLinearSVCClassifierForHeart():
    return GetJobLibFile(config['heart']['LinearSVC'])

def GetLogisticRegressionClassifierForHeart():
    return GetJobLibFile(config['heart']['LogisticRegression'])

def GetRandomForestClassifierForHeart():
    return GetJobLibFile(config['heart']['RandomForest'])

def GetKNeighborsClassifierForHeart():
    return GetJobLibFile(config['heart']['KNeighbors'])

def GetGradientBoastingClassifierForHeart():
    return GetJobLibFile(config['heart']['GradientBoasting'])

def GetEnsembleClassifierForHeart():
    return GetJobLibFile(config['heart']['Ensemble'])

## Diabetes

def GetAllClassifiersForDiabetes():
    return (GetLinearSVCClassifierForDiabetes(), GetLogisticRegressionClassifierForDiabetes(), GetRFClassifierForDiabetes(), GetKNeighborsClassifierForDiabetes(), GetGBClassifierForDiabetes(), GetEnsembleClassifierForDiabetes())

def GetStandardScalarForDiabetes():
    return GetPickleFile(config['diabetes']['scalar_file'])

def GetLinearSVCClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['LinearSVC'])

def GetLogisticRegressionClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['LogisticRegression'])

def GetRFClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['RandomForest'])

def GetKNeighborsClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['KNeighbors'])

def GetGBClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['GradientBoasting'])

def GetEnsembleClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['Ensemble'])

# def GetGradientClassifierForDiabetes():
#     return GetJobLibFile(config['diabetes']['Gradient'])

