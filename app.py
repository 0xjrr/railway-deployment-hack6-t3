import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    admission_id = IntegerField(unique=True)
    observation = TextField()
    readmitted = TextField()
    actual_readmitted = TextField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Begin Checks

numerical_values_minmax = {'admission_type_code': [
                                                    1.0,
                                                    8.0],
                             'discharge_disposition_code': [
                                                    1.0,
                                                    28.0],
                             'time_in_hospital': [
                                                    1,
                                                    14],
                             'num_lab_procedures': [
                                                    1.0,
                                                    132.0],
                             'num_procedures': [
                                                    0,
                                                    6],
                             'num_medications': [
                                                    1.0,
                                                    70.0],
                             'number_outpatient': [
                                                    0,
                                                    35],
                             'number_emergency': [
                                                    0,
                                                    22],
                             'number_inpatient': [
                                                    0,
                                                    18],
                             'number_diagnoses': [
                                                    1,
                                                    16],
                             'hemoglobin_level': [
                                                    10.9,
                                                    18.1]}

def check_categorical_values(observation):
    numerical_features = ['time_in_hospital','number_inpatient','discharge_disposition_code','number_diagnoses','num_medications','num_procedures','number_emergency','number_outpatient','hemoglobin_level','num_lab_procedures',"admission_type_code"]
    for s in numerical_features :
        th = observation.get(s)
        if th is None:
            error = "Fieldmissing"
            return False, error
        if th < 0 :
            error = "Field is not between 0 and 20"
            return False, error
    
    return True, ""


def check_number_inpatient(observation):
    n_i = observation.get("number_inpatient")
    if n_i is None: # if n_i is None
        error = "Field `number_inpatient` missing"
        return False, error
    if not isinstance(n_i, int):
        error = "Field `number_inpatient` is not an integer"
        return False, error
    if n_i < 0 or n_i > 20:
        error = "Field `number_inpatient` is not between 0 and 20"
        return False, error
    return True, ""
def check_num_lab_procedures(observation):
    n_l_p = observation.get("num_lab_procedures")
    if n_l_p is None:
        error = "Field `num_lab_procedures` missing"
        return False, error
    if not isinstance(n_l_p, float):
        error = "Field `num_lab_procedures` is not an integer"
        return False, error
    if n_l_p < 1 or n_l_p > 150:
        error = "Field `num_lab_procedures` is not between 0 and 150"
        return False, error
    return True, ""
def check_time_in_hospital(observation):
    th = observation.get("time_in_hospital")
    if th is None:
        error = "Field `time_in_hospital` missing"
        return False, error
    if not isinstance(th, int):
        error = "Field `time_in_hospital` is not an integer"
        return False, error
    if th < 1 or th > 20:
        error = "Field `time_in_hospital` is not between 0 and 20"
        return False, error
    return True, ""
def check_discharge_disposition_code(observation):
    dpc = observation.get("discharge_disposition_code")
    if dpc is None:
        error = "Field `discharge_disposition_code` missing"
        return False, error
    if not isinstance(dpc, float):
        error = "Field `discharge_disposition_code` is not an integer"
        return False, error
    if dpc < 1 or dpc > 30:
        error = "Field `discharge_disposition_code` is not between 0 and 30"
        return False, error
    return True, ""
def check_categorical_values(observation):
    valid_category_map = {
        "blood_type": ['A-', 'O+', 'A+', 'B+', 'O-', 'AB-', 'AB+', 'B-'],
        "insulin": ['Yes', 'No'],
        "age" : ['[80-90)', '[40-50)', '[30-40)', '[70-80)', '[90-100)', '[50-60)','[60-70)', '[20-30)', '[10-20)', '[0-10)'],
        "gender" : ['Male', 'Female', 'Unknown/Invalid'],
        "race" : ['Caucasian', 'European', 'AfricanAmerican', 'Other', 'Asian','Black', 'Hispanic', 'Latino'],
        "diabetesMed": ['Yes', 'No'],
        "complete_vaccination_status": ['Complete', 'Incomplete', 'None']
    }
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error
    return True, ""

# End Checks
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json.
    obs_dict: dict = request.get_json()
    ########################################
    # tests
    try:
        obs = pd.DataFrame([observation], columns=observation.keys())[columns].astype(dtypes)

        df_clean = obs.copy()
        df_clean['race'] = df_clean['race'].replace(['AFRICANAMERICAN', 'Afro American','African American'], 'AfricanAmerican')
        df_clean['race'] = df_clean['race'].replace(['EURO'], 'European')
        df_clean['race'] = df_clean['race'].replace(['WHITE', 'White'], 'Caucasian')
        df_clean['max_glu_serum'] = df_clean['max_glu_serum'].replace(['NONE'], 'None')
        df_clean['max_glu_serum'] = df_clean['max_glu_serum'].replace(['NORM'], 'Norm')
        df_clean['change'] = df_clean['change'].replace(['Ch'], 'Yes')
        df_clean['weight'] = df_clean['weight'].fillna('?')
        df_clean['age'] = df_clean['age'].fillna('?')



        _id = obs_dict['admission_id']
        observation = obs_dict
    except:
        categories_ok, error = check_categorical_values(obs_dict)
        if not categories_ok:
            response = {'error': error}
            return jsonify(response)
        cdp, error = check_discharge_disposition_code(obs_dict)
        if not cdp:
            response = {'error': error}
            return jsonify(response)
        ctih, error = check_time_in_hospital(obs_dict)
        if not ctih:
            response = {'error': error}
            return jsonify(response)
        cnlp, error = check_num_lab_procedures(obs_dict)
        if not cnlp:
            response = {'error': error}
            return jsonify(response)
        cni, error = check_number_inpatient(obs_dict)
        if not cni:
            response = {'error': error}
            return jsonify(response)
        
        _id = obs_dict['admission_id']
        observation = obs_dict
        
        obs = pd.DataFrame([observation], columns=observation.keys())[columns].astype(dtypes)
        df_clean = obs.copy()
        df_clean['race'] = df_clean['race'].replace(['AFRICANAMERICAN', 'Afro American','African American'], 'AfricanAmerican')
        df_clean['race'] = df_clean['race'].replace(['EURO'], 'European')
        df_clean['race'] = df_clean['race'].replace(['WHITE', 'White'], 'Caucasian')
        df_clean['max_glu_serum'] = df_clean['max_glu_serum'].replace(['NONE'], 'None')
        df_clean['max_glu_serum'] = df_clean['max_glu_serum'].replace(['NORM'], 'Norm')
        df_clean['change'] = df_clean['change'].replace(['Ch'], 'Yes')
        df_clean['weight'] = df_clean['weight'].fillna('?')
        df_clean['age'] = df_clean['age'].fillna('?')



    
    
    
    # Now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline.
    
    
    # Now get ourselves an actual prediction of the positive class.
    # prediction = pipeline.predict(df_clean)[0]
    y_pred_proba = pipeline.predict_proba(df_clean)[0,1]
    prediction_value = "Yes" if y_pred_proba>=0.47 else "No"
    prediction = prediction_value
    response = {'readmitted': prediction}
    p = Prediction(
        admission_id=_id,
        readmitted=prediction,
        observation= str(observation)
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.admission_id == obs['admission_id'])
        p.actual_readmitted = obs['readmitted']
        p.save()
        model_dict = model_to_dict(p)
        model_response_dict = {}
        model_response_dict["admission_id"] = model_dict["admission_id"]
        model_response_dict["actual_readmitted"] = model_dict["actual_readmitted"]
        model_response_dict["predicted_readmitted"] = model_dict["readmitted"]
        return jsonify(model_response_dict)
    except Prediction.DoesNotExist:
        error_msg = 'admission_id: "{}" does not exist'.format(obs['admission_id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
