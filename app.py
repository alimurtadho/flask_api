from flask import Flask, request, jsonify, make_response
from dataprovider import *
from wtforms import Form, IntegerField, SelectField, DecimalField, validators

app = Flask(__name__)

class HeartDiseaseForm(Form):
    age = IntegerField('Age', [validators.required()])
    sex = SelectField(choices=[('0', 'Female'),('1', 'Male')])
    cp = SelectField(choices=[('1', 'Typical Angina'),('2', 'Atypical Angina'),('3', 'Non-Angina'),('4', 'Asymptomatic')])
    resting_bp = IntegerField('ThrestBPS')
    serum_cholesterol = IntegerField('Serum Cholestrol')
    fasting_blood_sugar = IntegerField('Fasting Blood Sugar')
    resting_ecg = SelectField(choices=[('0', 'Normal'),('1', 'Having ST-T wave abnormality'),('2', 'Showing probable or definite left ventricular hupertrophy by Estes\' Criteria')])
    max_heart_rate = IntegerField('Thalach')
    exercise_induced_angina = SelectField(choices=[('0', 'No'),('1', 'Yes')])
    st_depression = DecimalField('Oldpeak')
    st_slope = SelectField(choices=[('1', 'Upsloping'),('2', 'Flat'),('3', 'Down Sloping')])
    number_of_vessels = SelectField(choices=[('0', 'None'),('1', 'One'),('2', 'Two'),('3', 'Three')])
    thallium_scan_results = SelectField(choices=[('3', 'Normal'),('6', 'Fixed Defect'),('7', 'Reversible Defect')])


@app.route('/predict/heart', methods=['POST'])
def PredictHeartDisease():
    form = HeartDiseaseForm(request.form)
    if form.validate():
        features = [[ form.age.data, form.sex.data, form.cp.data, form.resting_bp.data, form.serum_cholesterol.data, 0 if form.fasting_blood_sugar.data < 120 else 1, form.resting_ecg.data, form.max_heart_rate.data, form.exercise_induced_angina.data, form.st_depression.data, form.st_slope.data, form.number_of_vessels.data, form.thallium_scan_results.data ]]
        standard_scalar = GetStandardScalarForHeart()
        features_standard = standard_scalar.transform(features)

        LinearSVCClassifier, LogisticRegressionClassifier, NaiveBayesClassifier, KNeighborsClassifier, NeuralNetworkClassifier, EnsembleClassifier = GetAllClassifiersForHeart()

        predictions = {'Ensemble': str(EnsembleClassifier.predict(features_standard)[0]), 'LinearSVC': str(LinearSVCClassifier.predict(features_standard)[0]), 'LogisticRegression': str(LogisticRegressionClassifier.predict(features_standard)[0]), 'NaiveBayes': str(NaiveBayesClassifier.predict(features_standard)[0]), 'KNeighbors': str(KNeighborsClassifier.predict(features_standard)[0]), 'NeuralNetwork': str(NeuralNetworkClassifier.predict(features_standard)[0]) }
        return make_response(jsonify(predictions), 200)
    else:
        return make_response(jsonify({'errors': form.errors}), 400)

class DiabetesForm(Form):
    age = IntegerField('Age', [validators.required()])
    pregnant = IntegerField('Number of times pregnant')
    plasma_glucose_concentration = IntegerField('Plasma Glucose Concentration')
    diastolic_bp = IntegerField('Diastolic BP')
    tsft = IntegerField('tsft')
    serum_insulin = IntegerField('Serum Insulin')
    bmi = DecimalField('BMI')
    dpf = DecimalField('DFP')



@app.route('/predict/diabetes', methods=['POST'])
def PredictDiabetes():
    form = DiabetesForm(request.form)
    if form.validate():
        features = [[ form.pregnant.data, form.plasma_glucose_concentration.data, form.diastolic_bp.data, form.tsft.data, form.serum_insulin.data, form.bmi.data, form.dpf.data, form.age.data ]]
        standard_scalar = GetStandardScalarForDiabetes()
        features_standard = standard_scalar.transform(features)

        LinearSVCClassifier, LogisticRegressionClassifier, NaiveBayesClassifier, KNeighborsClassifier, NeuralNetworkClassifier, EnsembleClassifier = GetAllClassifiersForDiabetes()

        predictions = {'Ensemble': str(EnsembleClassifier.predict(features_standard)[0]), 'LinearSVC': str(LinearSVCClassifier.predict(features_standard)[0]), 'LogisticRegression': str(LogisticRegressionClassifier.predict(features_standard)[0]), 'NaiveBayes': str(NaiveBayesClassifier.predict(features_standard)[0]), 'KNeighbors': str(KNeighborsClassifier.predict(features_standard)[0]), 'NeuralNetwork': str(NeuralNetworkClassifier.predict(features_standard)[0]) }
        return make_response(jsonify(predictions), 200)
    else:
        return make_response(jsonify({'errors': form.errors}), 400)


if __name__ == "__main__":
    app.run()