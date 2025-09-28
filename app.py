# Add render_template to the first line
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS to allow requests from the frontend
CORS(app)

# --- Load Trained Model and Supporting Files ---
try:
    print("Loading model and required files...")
    lgb_model = joblib.load('lgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print("Files loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found! Please run train_model.py first.")
    exit()

# --- NEW: ADD A ROUTE TO SERVE THE HTML PAGE ---
@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')
# ---------------------------------------------

# --- Define the Prediction Function ---
def create_features_and_predict(data):
    """
    Takes a dictionary of input data, engineers features, and returns a prediction.
    """
    # Extract data from the request
    date_str = data.get('date', '2025-09-18')
    time_str = data.get('time', '18:00')
    temp_celsius = float(data.get('temp', 25.0))
    humidity_percent = float(data.get('humidity', 60.0))
    windspeed_kmh = float(data.get('windspeed', 10.0))
    weathersit = int(data.get('weathersit', 1))
    
    # Process date and time
    dt_obj = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')
    
    # Normalize weather inputs
    temp = (temp_celsius - (-8)) / (39 - (-8))
    hum = humidity_percent / 100
    windspeed = windspeed_kmh / 67

    # --- Feature Engineering ---
    season_map = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
    season = season_map[dt_obj.month]
    yr = dt_obj.year - 2011
    mnth = dt_obj.month
    hr = dt_obj.hour
    weekday = (dt_obj.weekday() + 1) % 7 # Convert from Mon=0 to Sun=0
    holiday = 1 if weekday in [0, 6] else 0
    workingday = 1 - holiday

    if 5 <= hr <= 11: part_of_day = 'Morning'
    elif 12 <= hr <= 16: part_of_day = 'Afternoon'
    elif 17 <= hr <= 21: part_of_day = 'Evening'
    else: part_of_day = 'Night'

    weather_day_interaction = f"{weathersit}_{part_of_day}"
    heat_index = temp * hum
    hr_sin = np.sin(2 * np.pi * hr / 24.0)
    hr_cos = np.cos(2 * np.pi * hr / 24.0)
    mnth_sin = np.sin(2 * np.pi * (mnth - 1) / 12.0)
    mnth_cos = np.cos(2 * np.pi * (mnth - 1) / 12.0)
    weekday_sin = np.sin(2 * np.pi * weekday / 7.0)
    weekday_cos = np.cos(2 * np.pi * weekday / 7.0)

    # Create DataFrame for prediction
    input_data = {
        'temp': [temp], 'hum': [hum], 'windspeed': [windspeed],
        'hr_sin': [hr_sin], 'hr_cos': [hr_cos],
        'mnth_sin': [mnth_sin], 'mnth_cos': [mnth_cos],
        'weekday_sin': [weekday_sin], 'weekday_cos': [weekday_cos],
        'heat_index': [heat_index],
        'season': [season], 'yr': [yr], 'holiday': [holiday],
        'workingday': [workingday], 'weathersit': [weathersit],
        'part_of_day': [part_of_day], 'weather_day_interaction': [weather_day_interaction]
    }
    input_df = pd.DataFrame(input_data)
    input_df_processed = pd.get_dummies(input_df)
    input_df_aligned = input_df_processed.reindex(columns=model_columns, fill_value=0)
    
    numerical_features = [
        'temp', 'hum', 'windspeed', 'hr_sin', 'hr_cos',
        'mnth_sin', 'mnth_cos', 'weekday_sin', 'weekday_cos',
        'heat_index'
    ]
    input_df_aligned[numerical_features] = scaler.transform(input_df_aligned[numerical_features])

    # --- Prediction ---
    prediction_log = lgb_model.predict(input_df_aligned)
    prediction = np.expm1(prediction_log)
    
    return int(np.round(prediction[0]))

# --- Define API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        prediction_result = create_features_and_predict(data)
        return jsonify({'prediction': prediction_result})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)