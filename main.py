from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

selected_features = ['Age', 'Income', 'Dependents', 'Education', 'Married', 'Credit_History', 'Credit_mix', 'Inactivity', 'Inquiries', 'Credit_Limit', 'Balance', 'Yearly_Amount_Change', 'Spent', 'Utilization', 'Credit_Score', 'Delinquency_Ratio', 'Delinquency', 'Spender']

# Load your credit score and limit models
model_credit_score = load_model('C:\\Users\\spsam\\Desktop\\Credit app\\model_credit_score.h5')
model_credit_limit = load_model('C:\\Users\\spsam\\Desktop\\Credit app\\model_credit_limit.h5')

# Statistical values for normalization
mean_age = 30.0
std_age = 5.0
mean_income = 50000.0
std_income = 10000.0

# Statistical values for reverse normalization
min_age, max_age = 20.0, 70.0
min_dependents, max_dependents = 0.0, 5.0
# Add other min-max values for reverse normalization as needed

# Constants for reverse normalization
min_credit_score = 300
max_credit_score = 850
min_credit_limit = 1000
max_credit_limit = 100000

# Function for preprocessing input data
def preprocess_input_data(CLIENTNUM,age, income, Dependents, Education, Married, Credit_History, Credit_mix, Inactivity, Inquiries, Credit_Limit, Balance, Yearly_Amount_Change, Spent, Utilization, Credit_Score, Delinquency_Ratio, Delinquency, Spender):
    # Apply the same preprocessing steps used during training
    standardized_age = (age - mean_age) / std_age
    standardized_income = (income - mean_income) / std_income
    # Apply the same preprocessing steps for other features

    return np.array([[CLIENTNUM,standardized_age, standardized_income, Dependents, Education, Married, Credit_History, Credit_mix, Inactivity, Inquiries, Credit_Limit, Balance, Yearly_Amount_Change, Spent, Utilization, Credit_Score, Delinquency_Ratio, Delinquency, Spender]])

# Function for reverse normalization of credit score
def reverse_normalize_credit_score(credit_score):
    # Apply the reverse normalization process
    actual_credit_score = credit_score * (max_credit_score - min_credit_score) + min_credit_score
    return actual_credit_score

# Function for reverse normalization of credit limit
def reverse_normalize_credit_limit(credit_limit):
    # Apply the reverse normalization process
    actual_credit_limit = credit_limit * (max_credit_limit - min_credit_limit) + min_credit_limit
    return actual_credit_limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    CLIENTNUM = float(request.form.get('CLIENTNUM', 0))
    age = float(request.form.get('age', 0))
    income = float(request.form.get('income', 0))
    # Add other input features as needed
    dependents = float(request.form.get('dependents', 0))
    education = float(request.form.get('education', 0))
    married = float(request.form.get('married', 0))
    credit_history = float(request.form.get('credit_history', 0))
    credit_mix = float(request.form.get('credit_mix', 0))
    inactivity = float(request.form.get('inactivity', 0))
    inquiries = float(request.form.get('inquiries', 0))
    credit_limit = float(request.form.get('credit_limit', 0))
    balance = float(request.form.get('balance', 0))
    yearly_amount_change = float(request.form.get('yearly_amount_change', 0))
    spent = float(request.form.get('spent', 0))
    utilization = float(request.form.get('utilization', 0))
    credit_score = float(request.form.get('credit_score', 0))
    delinquency_ratio = float(request.form.get('delinquency_ratio', 0))
    delinquency = float(request.form.get('delinquency', 0))
    spender = float(request.form.get('spender', 0))

    # Preprocess input data
    input_data = preprocess_input_data(CLIENTNUM,age, income, dependents, education, married, credit_history, credit_mix, inactivity, inquiries, credit_limit, balance, yearly_amount_change, spent, utilization, credit_score, delinquency_ratio, delinquency, spender)

    # Make predictions
    credit_score_result = model_credit_score.predict(input_data)[0][0]
    credit_limit_result = model_credit_limit.predict(input_data)[0][0]

 # Reverse normalization
    credit_score_result = reverse_normalize_credit_score(credit_score_result)
    credit_limit_result = reverse_normalize_credit_limit(credit_limit_result)


    return render_template('results.html', credit_score=credit_score_result, credit_limit=credit_limit_result)

if __name__ == '__main__':
    app.run(debug=True)
