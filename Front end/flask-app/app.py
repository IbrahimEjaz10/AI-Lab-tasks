from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv('student_exam_data_new.csv')
X = data[['Study Hours', 'Previous Exam Score']]
y = data['Pass/Fail']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    study_hours = float(request.form['study_hours'])
    previous_exam_score = float(request.form['previous_exam_score'])
    
    
    user_input_scaled = scaler.transform([[study_hours, previous_exam_score]])
    user_prediction = model.predict(user_input_scaled)
    
    result = "Pass" if user_prediction[0] == 1 else "Fail"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)