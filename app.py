from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load scaler and model
with open("diabetes_model.pkl", "rb") as f:
    scaler, model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    input_values = [float(x) for x in request.form.values()]
    input_array = np.array([input_values])

    # Scale input and predict
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    # Prepare result and advice
    if prediction == 1:
        result = "Diabetic"
        advice = (
            "You are predicted to be diabetic. Please consult a healthcare professional, "
            "maintain a balanced diet, exercise regularly, and monitor your blood sugar levels."
        )
    else:
        result = "Not Diabetic"
        advice = (
            "You are not predicted to be diabetic. Maintain a healthy lifestyle with a balanced diet "
            "and regular exercise to keep yourself fit."
        )

    return render_template("index.html", prediction_text=f"Prediction: {result}", advice_text=advice)

if __name__ == "__main__":
    app.run(debug=True)
