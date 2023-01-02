from flight_fare.credentials import *
from flight_fare.utils import transform, predict_fare
from flight_fare.logger import logging
from flight_fare.exception import FlightFareException
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app = Flask(__name__)

@app.route('/')
def home():
    logging.info(f"Rendering Home Page")
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    logging.info(f"Collecting data from API")
    data=request.json['data']
    logging.info(f"Converting collected data into 2-D array.")
    new_data = np.array(list(data.values())).reshape(1,-1)
    logging.info("Predicting Flight Fare")
    prediction = predict_fare(new_data)
    logging.info(f"Displayed Predicted Fare")
    logging.info(prediction[0])
    return jsonify(prediction[0])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        logging.info(f"Collecting data from Web page")
        data = [x for x in request.form.values()]

        logging.info(f"Collected Info : {data}")

        trf_data = transform(data)

        # converting the transformed list into a DataFrame with all feature names as column names
        model_input = pd.DataFrame(columns=columns)
        model_input.loc[len(model_input.index)] = trf_data
        logging.info(f"Input DataFrame shape : {model_input.shape}")

        result = predict_fare(model_input)
        
        prediction_text = f"Predicted Flight Fare is INR {result}"
        logging.info(f"Final result : {prediction_text}")

        return render_template("home.html", prediction_text=prediction_text)
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
