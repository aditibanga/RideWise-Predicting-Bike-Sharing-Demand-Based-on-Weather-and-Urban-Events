import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib

def preprocess_and_train(df_hour):
   
    # --- Feature Engineering ---
    print("Starting feature engineering...")
   
    def get_part_of_day(hour):
        if 5 <= hour <= 11: return 'Morning'
        elif 12 <= hour <= 16: return 'Afternoon'
        elif 17 <= hour <= 21: return 'Evening'
        else: return 'Night'
    df_hour['part_of_day'] = df_hour['hr'].apply(get_part_of_day)

    # Weather and day interaction
    df_hour['weather_day_interaction'] = df_hour['weathersit'].astype(str) + "_" + df_hour['part_of_day']

    # Heat Index
    df_hour['heat_index'] = df_hour['temp'] * df_hour['hum']

    # Cyclical time features
    df_hour['hr_sin'] = np.sin(2 * np.pi * df_hour['hr'] / 24.0)
    df_hour['hr_cos'] = np.cos(2 * np.pi * df_hour['hr'] / 24.0)
    df_hour['mnth_sin'] = np.sin(2 * np.pi * (df_hour['mnth'] - 1) / 12.0)
    df_hour['mnth_cos'] = np.cos(2 * np.pi * (df_hour['mnth'] - 1) / 12.0)
    df_hour['weekday_sin'] = np.sin(2 * np.pi * df_hour['weekday'] / 7.0)
    df_hour['weekday_cos'] = np.cos(2 * np.pi * df_hour['weekday'] / 7.0)
    print("Feature engineering complete.")

    # --- Preprocessing ---
    print("Starting data preprocessing...")
    # Log transform the target variable
    df_hour['cnt'] = np.log1p(df_hour['cnt'])

    # One-hot encode categorical features
    categorical_features = ['season', 'yr', 'holiday', 'workingday', 'weathersit',
                            'part_of_day', 'weather_day_interaction']
    df_processed = pd.get_dummies(df_hour, columns=categorical_features, drop_first=True)

    # Drop original and unnecessary columns
    columns_to_drop = ['instant', 'dteday', 'atemp', 'casual', 'registered',
                       'hr', 'mnth', 'weekday']
    df_processed = df_processed.drop(columns=columns_to_drop, axis=1)

    # Separate features (X) and target (y)
    X = df_processed.drop('cnt', axis=1)
    y = df_processed['cnt']

    # Split data chronologically
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows).")

    # Scale numerical features
    numerical_features = [
        'temp', 'hum', 'windspeed', 'hr_sin', 'hr_cos',
        'mnth_sin', 'mnth_cos', 'weekday_sin', 'weekday_cos',
        'heat_index'
    ]
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    print("Numerical features scaled.")

    # --- Model Training ---
    print("Training LightGBM model...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    print("Model training complete.")

  
    print("Saving model, scaler, and column list...")
    joblib.dump(lgb_model, 'lgb_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')
    print("Files saved successfully.")

    return lgb_model, scaler, X_test, y_test

if __name__ == '__main__':
    try:
        df = pd.read_csv('hour.csv')
        preprocess_and_train(df)
    except FileNotFoundError:
        print("Error: hour.csv not found. Please place it in the same directory.")
