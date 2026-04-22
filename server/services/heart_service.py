import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_MODELS_DIR = Path("saved_models")
_HEART_BUNDLE = _MODELS_DIR / "heart_bundle.pkl"

preprocessor_transformer_heart = None
model_trained_heart = None

def train_heart_model():
    global preprocessor_transformer_heart, model_trained_heart
    _MODELS_DIR.mkdir(exist_ok=True)
    if _HEART_BUNDLE.exists():
        b = joblib.load(_HEART_BUNDLE)
        preprocessor_transformer_heart, model_trained_heart = b["preprocessor"], b["model"]
        logger.info("Heart model loaded from disk.")
        return
    try:
        data = pd.read_csv('diseaseheart/heart_disease_data.csv')  # local

    except Exception as e:
        logger.error(f"Failed to load Heart Disease Dataset: {e}")
        return

    # Ensure the dataset includes all 14 fields
    expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    if not all(col in data.columns for col in expected_columns):
        raise ValueError('Dataset does not have the expected columns.')

    # Prepare features and target variable
    # NOTE: We use the raw data (with string categories) because ColumnTransformer will handle OHE.
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Identify categorical and numerical features
    data_cat = X.select_dtypes("object")
    data_num = X.select_dtypes(["int64", "float64"])
    
    num_features = data_num.columns.tolist()
    cat_features = data_cat.columns.tolist()
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_features)
        ]
    )
    
    # Apply preprocessing
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = tts(X_preprocessed, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Save transformers for prediction
    preprocessor_transformer_heart = preprocessor
    model_trained_heart = model
    joblib.dump({"preprocessor": preprocessor, "model": model}, _HEART_BUNDLE)
    logger.info("Heart model trained and saved.")

def predict_heart_disease(input_data_dict):
    global preprocessor_transformer_heart, model_trained_heart
    if model_trained_heart is None:
        train_heart_model()
        
    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data_dict])
    
    # Ensure all columns are present (fill missing numeric with 0, cat with '')
    # We should match X columns.
    # The ColumnTransformer is robust to column order if passed a DataFrame, but we need to ensure keys exist.
    
    # We rely on input_data_dict having the keys.
    # The frontend usually sends specific fields.
    
    try:
        # Apply save transformers to the input data
        # input_df should contain the RAW values (strings for sex, cp, etc.)
        input_data_preprocessed = preprocessor_transformer_heart.transform(input_df)
        
        # Prediction & probability
        prediction = model_trained_heart.predict(input_data_preprocessed)
        probabilities = model_trained_heart.predict_proba(input_data_preprocessed)
        
        # Probability of Class 1 (Disease)
        prob_percent = probabilities[0][1] * 100
        
        return int(prediction[0]), f"{prob_percent:.2f}"
    except Exception as e:
        logger.error(f"Prediction failed in heart service: {e}")
        raise e
