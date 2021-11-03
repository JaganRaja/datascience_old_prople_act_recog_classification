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
def homePage():
    return render_template("index.html")


@app.route('/predict_activity', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            #json_ = request.json

            # reading the inputs given by the user
            time = float(request.form['time(s)'])
            frontal_axis = float(request.form['frontal_axis'])
            vertical_axis = float(request.form['vertical_axis'])
            lateral_axis = float(request.form['lateral_axis'])
            antenna_id = float(request.form['antenna_id'])
            rssi = float(request.form['rssi'])
            phase = float(request.form['phase'])
            frequency = float(request.form['frequency'])
            gender = request.form['gender']

            # making the iput as dictionary
            # key --> column name
            # value --> user given value
            value_dict = {
                "time(s)": time,
                "frontal_axis": frontal_axis,
                "vertical_axis": vertical_axis,
                "lateral_axis": lateral_axis,
                "antenna_id": antenna_id,
                "rssi": rssi,
                "phase": phase,
                "frequency": frequency,
                "gender": gender
            }

            # load as dataframe
            query_df = pd.DataFrame([value_dict])

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
            print('prediction is', pred_val)
            # return jsonify(int(activity_prediction[0]))
            # return jsonify({'activity_prediction': int(activity_prediction[0])})
            # return jsonify({'activity_prediction': {pred_val: activity_label[pred_val]}})
            # return ({'activity_prediction': {pred_val: activity_label[pred_val]}})

            # showing the prediction results in a UI
            return render_template('results.html', prediction=activity_label[pred_val])
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
            1: "Sit on bed",
            2: "Sit on chair",
            3: "Lying",
            4: "Ambulating"
        }
    except Exception as e:
        print('The Exception message is: ', e)
        print('Error in loading model')

    app.run(debug=True)  # running the app
