# importing the necessary dependencies

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

# refer
# https://www.datacamp.com/community/tutorials/machine-learning-models-api-python

# initializing a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def hello():
    return "Hello, World!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"


@app.route('/predict_activity', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            json_ = operation = request.json

            # load as dataframe
            query_df = pd.DataFrame([json_])

            # data preprocessing - transform the new data
            # apply encoding to 'gender' column
            query_df['gender'] = enoder_obj.transform(query_df[['gender']])
            # apply StandardScaler to features
            query_df_scaled = standard_scaler_obj.transform(query_df)
            # predict
            activity_prediction = model.predict(query_df_scaled)
            # activity_prediction - datatype will be numpy.int64, so we need to change to python datatype int,str,list etc..
            pred_val = int(activity_prediction[0])

            # display the results
            # return jsonify(int(activity_prediction[0]))
            # return jsonify({'activity_prediction': int(activity_prediction[0])})
            return jsonify({'activity_prediction': {pred_val: activity_label[pred_val]}})
            # return ({'activity_prediction': {pred_val: activity_label[pred_val]}})
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'


if __name__ == '__main__':

    try:
        # Load "people_activity_recog_classification.sav"
        model = pickle.load(
            open("people_activity_recog_classification.pkl", 'rb'))
        print('Model loaded')

        # Load "encoder_gender.sav"
        enoder_obj = pickle.load(open("encoder_gender.pkl", 'rb'))
        print('encoder loaded')

        # Load "standard_scaler.sav"
        standard_scaler_obj = pickle.load(open("standard_scaler.pkl", 'rb'))
        print('standard scaler loaded')

        activity_label = {
            1: "sit on bed",
            2: "sit on chair",
            3: "lying",
            4: "ambulating"
        }
    except Exception as e:
        print('The Exception message is: ', e)
        print('Error in loading model')

    app.run(debug=True)  # running the app
