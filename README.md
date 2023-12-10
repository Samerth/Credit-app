# Credit-app

Credit Prediction Web Application
This web application utilizes neural network models to predict credit scores and credit limits based on user-provided information. The models are trained on a preprocessed dataset, and the Flask framework is used to create a user-friendly interface.

Setup and Installation
Clone the repository to your local machine:


Run the Flask application:


python app.py
Open your web browser and go to http://localhost:5000/ to access the web app.

Web App Features
Input Form: Fill out the form with relevant information for credit prediction.
Prediction Results: View the predicted credit score and credit limit.
User-Friendly Interface: The web app is designed for easy and intuitive use.
Model Information
Model Architecture: Two neural network models are used – one for credit score prediction and another for credit limit prediction.
Metrics: The models are evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).
Repository Structure
app.py: Flask application script.
templates/: HTML templates for the web app.
static/: Static files (CSS, images, etc.).
models/: Directory to store trained model files.
Additional Notes
The web app uses StandardScaler for feature standardization during model training and prediction.
Reverse normalization functions are implemented to convert predicted values back to their original scales.
Contributors
Samerth Pathak
