import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import GradientBoostingClassifier
import logging

logger = logging.getLogger(__name__)

gbm_model_liver = None
scaler_liver = None

def train_liver_model():
    global gbm_model_liver, scaler_liver
    # Use local file path relative to this service file? 
    # Current dir: server/services. File: server/liver/data.csv
    # Better to use absolute or relative to 'server' root if run from app.py.
    # app.py runs in 'server' dir. So 'liver/Liver_disease_data.csv' works.
    try:
        Liver_DS = pd.read_csv('liver/Liver_disease_data.csv')
    except Exception as e:
        logger.error(f"Failed to load Liver Dataset locally: {e}. Trying URL fallback.")
        try:
             Liver_DS = pd.read_csv('https://raw.githubusercontent.com/ARSHIYASHAFIZADE/SAM_Ai/refs/heads/main/server/liver/Liver_disease_data.csv')
        except Exception as e2:
             logger.error(f"Failed URL fallback: {e2}")
             return

    # Handle missing values first (imputation)
    Liver_DS = Liver_DS.fillna(0) # Simple imputation to prevent crashes
    
    # Handle Categorical 'Gender'
    # Check if Gender is string
    if Liver_DS['Gender'].dtype == 'object':
        Liver_DS['Gender'] = Liver_DS['Gender'].map({'Male': 0, 'Female': 1})
    
    # Ensure Gender is numeric even if mapping failed (e.g. fill na)
    Liver_DS['Gender'] = pd.to_numeric(Liver_DS['Gender'], errors='coerce').fillna(0)

    # Fix drop syntax: cannot use both columns and axis
    X = Liver_DS.drop(columns=['Diagnosis'])
    Y = Liver_DS['Diagnosis']   
    
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        logger.error(f"Scaling failed: {e}. Check for non-numeric columns.")
        return

    x_train, x_va, y_train, y_va = tts(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
    
    gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.121, random_state=103)
    gbm_model.fit(x_train, y_train)
    
    gbm_model_liver = gbm_model
    scaler_liver = scaler

def predict_liver_health(input_data):
    if gbm_model_liver is None:
        train_liver_model()
    
    if scaler_liver is None:
        raise ValueError("Model training failed. Scaler is None.")

    # Input data comes from Frontend as an array of 10 values [Age, Gender, Bilirubin, ...]
    # Model expects 10 features [Age, Gender, BMI, Alcohol, ...]
    # We must ensure the shape matches.
    # Fortunately, counts match (10 vs 10).
    # But we need to ensure they are all numbers.
    
    try:
        # Cast to float array to handle any string "0" or "1"
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)
        input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
        
        # Check shape
        if input_data_reshape.shape[1] != 10:
             # If mismatch, we might need to pad or trim?
             # But frontend sends 10. Dataset has 10. Should be fine.
             logger.warning(f"Input shape {input_data_reshape.shape} != 10. Prediction might fail.")

        standardized_input_data = scaler_liver.transform(input_data_reshape)
        
        probabilities = gbm_model_liver.predict_proba(standardized_input_data)[0]
        prediction = np.argmax(probabilities)
        
        # Return Class 0 (Healthy) probability
        prob_percent = probabilities[0] * 100
        probability_healthy_liver = f"{prob_percent:.2f}%"
        
        return int(prediction), probability_healthy_liver
    except Exception as e:
         logger.error(f"Prediction failed: {e}")
         raise e
