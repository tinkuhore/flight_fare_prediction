import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app = Flask(__name__)
# load model
gen_model = pickle.load(open("pkl_files/gen_pred_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    data=request.json['data']
    new_data = np.array(list(data.values())).reshape(1,-1)
    prediction = gen_model.predict(new_data)
    print(prediction[0])
    return jsonify(prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
