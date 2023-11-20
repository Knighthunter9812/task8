import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')
   # return 'hello world'

@app.route('/predict',methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    intpred=int(prediction)
    return render_template("index.html", prediction_text="The Test Results is {}".format(intpred))

if __name__=="__main__":
    app.run(debug=True)