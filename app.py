import pickle
import flask
from flask import Flask, request, app, jsonify, url_for, render_template # jsonify converts an output into a jason
from flask import Response
from flask_cors import CORS
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb")) # This is how we load a pickle file

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods = ["POST"]) # This is how you create an API
def predict_api():

    data = request.json["data"] # To capture json information coming from the postman, we use "request" library.
    print(data)
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]
    return jsonify(output)

@app.route("/predict", methods = ["POST"]) # This is how you create an API
def predict():

    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    output = model.predict(final_features)[0]
    print(output)
    return render_template("home.html", prediction_text = "Airfoil pressure is {}".format(output))

if __name__ == "__main__":
    app.run(debug = True)


