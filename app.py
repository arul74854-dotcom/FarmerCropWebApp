from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    values = [
        float(request.form['nitrogen']),
        float(request.form['phosphorus']),
        float(request.form['potassium']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['ph']),
        float(request.form['rainfall'])
    ]

    final_input = scaler.transform([values])

    probabilities = model.predict_proba(final_input)[0]
    classes = model.classes_

    sorted_idx = np.argsort(probabilities)[::-1]

    best_crop = classes[sorted_idx[0]]
    alt_crop = classes[sorted_idx[1]]

    confidence = round(probabilities[sorted_idx[0]] * 100, 2)

    probability_list = [round(p * 100, 2) for p in probabilities]

    explanation = []
    if values[6] > 150:
        explanation.append("High rainfall suitable for this crop")
    if 5.5 <= values[5] <= 7.5:
        explanation.append("Optimal soil pH level")
    if values[0] > 50:
        explanation.append("Good nitrogen content in soil")
    if not explanation:
        explanation.append("Prediction based on trained Machine Learning model")

    return render_template(
        "index.html",
        result=best_crop,
        confidence=confidence,
        alternative=alt_crop,
        explanation=explanation,
        labels=classes.tolist(),
        probabilities=probability_list
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


