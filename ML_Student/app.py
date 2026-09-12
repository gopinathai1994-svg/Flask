from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# load model
model = pickle.load(open('models/student_model.pkl', 'rb'))
scaler = pickle.load(open('models/student_scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    attendance = float(request.form['attendance'])
    monthly_mark = float(request.form['monthly_mark'])
    section = request.form['section']

    # Create a DataFrame with the input data
    data = pd.DataFrame({
        'Attendance': [attendance],
        'Monthly_Mark': [monthly_mark],
        'Section_B': [1 if section == 'B' else 0],
        'Section_C': [1 if section == 'C' else 0],
        'Section_D': [1 if section == 'D' else 0]
    })

    # Preprocess the data (e.g., encode categorical variables, scale features)
    # This step depends on how your model was trained
    data_processed = scaler.transform(data)
    
    # Make prediction
    prediction = model.predict(data_processed)
    print(prediction,'prediction')
    # Return the prediction result
    return render_template('result.html', prediction=prediction[0], attendance=attendance, monthly_mark=monthly_mark, section=section)  

if __name__ == "__main__":
    app.run(debug=True)