import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_MODELS_DIR = Path("saved_models")
_FEMALE_BUNDLE = _MODELS_DIR / "diabetes_female_bundle.pkl"
_MALE_BUNDLE   = _MODELS_DIR / "diabetes_male_bundle.pkl"

# Female Model Globals
transformer_female = None
scaler_female = None
gbc_female = None
model_female = None # Pipeline model
feature_names_female = None

# Male Model Globals
poly_male = None
scaler_male = None
lr_male = None
X_male_columns = None

def preprocess_female_diabetes():
    global transformer_female, scaler_female, gbc_female, model_female, feature_names_female
    _MODELS_DIR.mkdir(exist_ok=True)
    if _FEMALE_BUNDLE.exists():
        b = joblib.load(_FEMALE_BUNDLE)
        transformer_female, scaler_female, gbc_female, feature_names_female = (
            b["transformer"], b["scaler"], b["model"], b["feature_names"]
        )
        logger.info("Female diabetes model loaded from disk.")
        return
    try:
        Diabetes_DS = pd.read_csv('dfw.csv')  # local file in server/
    except Exception as e:
        logger.error(f"Failed to load Female Diabetes Dataset: {e}")
        return

    # Replace 0 values with NaN for relevant columns
    Diabetes_DS[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = Diabetes_DS[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.nan)

    # Function to get the median based on the target 'Outcome'
    def median_target(var):
        temp = Diabetes_DS[Diabetes_DS[var].notnull()]
        temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
        return temp

    # Fill missing values with median based on the 'Outcome' column
    columns = Diabetes_DS.columns.drop("Outcome")
    for i in columns:
        Diabetes_DS.loc[(Diabetes_DS['Outcome'] == 0) & (Diabetes_DS[i].isnull()), i] = median_target(i)[i][0]
        Diabetes_DS.loc[(Diabetes_DS['Outcome'] == 1) & (Diabetes_DS[i].isnull()), i] = median_target(i)[i][1]

    # Handle outliers in the Insulin column
    Q1 = Diabetes_DS.Insulin.quantile(0.25)
    Q3 = Diabetes_DS.Insulin.quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    Diabetes_DS.loc[Diabetes_DS['Insulin'] > upper, "Insulin"] = upper

    # Use Local Outlier Factor to detect and remove outliers
    lof = LocalOutlierFactor(n_neighbors=10)
    Diabetes_DS_scores = lof.fit_predict(Diabetes_DS.drop('Outcome', axis=1))
    threshold = np.sort(lof.negative_outlier_factor_)[7]
    outliers = Diabetes_DS_scores > threshold
    Diabetes_DS = Diabetes_DS[outliers]

    # Create new BMI categories
    NewBMI = ["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"]
    Diabetes_DS['NewBMI'] = pd.cut(Diabetes_DS['BMI'], bins=[-np.inf, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf], labels=NewBMI)

    # Create new Glucose categories
    NewGlucose = ["Low", "Normal", "Overweight", "High"]
    Diabetes_DS["NewGlucose"] = pd.cut(Diabetes_DS["Glucose"], bins=[-np.inf, 70, 99, 126, np.inf], labels=NewGlucose)

    # One-hot encode the new categorical columns
    Diabetes_DS = pd.get_dummies(Diabetes_DS, columns=["NewBMI", "NewGlucose"], drop_first=True)

    # Prepare features (X) and target (y)
    X = Diabetes_DS.drop(['Outcome'], axis=1)
    y = Diabetes_DS['Outcome']
    feature_names_female = X.columns.tolist()

    # Scale the features using RobustScaler
    transformer_female = RobustScaler().fit(X)
    X_scaled = transformer_female.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names_female, index=X.index)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = tts(X_scaled, y, test_size=0.2, random_state=0)

    # Standardize the training data
    scaler_female = StandardScaler()
    X_train_scaled = scaler_female.fit_transform(X_train)

    # Train a Gradient Boosting Classifier
    gbc_female = GradientBoostingClassifier(learning_rate=0.1, loss='exponential', n_estimators=150)
    gbc_female.fit(X_train_scaled, y_train)
    joblib.dump({"transformer": transformer_female, "scaler": scaler_female, "model": gbc_female, "feature_names": feature_names_female}, _FEMALE_BUNDLE)
    logger.info("Female diabetes model trained and saved.")

    # Train a Logistic Regression model in a pipeline
    model_lr = LogisticRegression(max_iter=1000)
    model_female = make_pipeline(scaler_female, model_lr) # Note: Creating pipeline, but not really using it as object.
    # The original code fit `model_lr` inside pipeline or `pipeline.fit`.
    # Let's replicate original logic:
    # pipeline.fit(X_train_scaled, y_train) -> This fits the scaler (if un-fitted) and model.
    # But wait, scaler_female was already fitted?
    # Actually, simpler: just store the objects that work.
    # We need scaler_female (StandardScaler) and gbc_female (Model) for prediction.
    # Original code used `model` (LR) for predict_proba too, but we standardized on `gbc` in the fix.
    # So we don't strictly need model_lr unless we want it. We'll stick to `gbc_female` for everything now.

def predict_female_diabetes(input_data):
    # Ensure model is trained
    if gbc_female is None:
        preprocess_female_diabetes()
        
    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])  
    
    # Apply preprocessing steps to the input data
    input_df['NewBMI'] = pd.cut(input_df['BMI'], bins=[-np.inf, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf], labels=["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"])
    input_df["NewGlucose"] = pd.cut(input_df["Glucose"], bins=[-np.inf, 70, 99, 126, np.inf], labels=["Low", "Normal", "Overweight", "High"])  

    # Generate dummies for categorical features
    input_df = pd.get_dummies(input_df, columns=["NewBMI", "NewGlucose"], drop_first=True)  
    
    # Reindex to match expected columns
    input_df = input_df.reindex(columns=feature_names_female, fill_value=0)  
    
    # Scale the input data using RobustScaler
    input_df_scaled = transformer_female.transform(input_df)  
    
    # Standardize using StandardScaler (gbc trained on standardized data)
    input_df_prepared = scaler_female.transform(input_df_scaled)
    
    # Make prediction (0 or 1)
    prediction = gbc_female.predict(input_df_prepared)
    
    # Probability (Class 1)
    probability = gbc_female.predict_proba(input_df_prepared)
    prob_percent = probability[0][1] * 100

    return int(prediction[0]), round(prob_percent, 2)


def preprocess_male_diabetes():
    global Diabetes_DS_male, poly_male, scaler_male, lr_male, X_male_columns
    _MODELS_DIR.mkdir(exist_ok=True)
    if _MALE_BUNDLE.exists():
        b = joblib.load(_MALE_BUNDLE)
        poly_male, scaler_male, lr_male, X_male_columns = (
            b["poly"], b["scaler"], b["model"], b["columns"]
        )
        logger.info("Male diabetes model loaded from disk.")
        return
    try:
        Diabetes_DS_male = pd.read_csv('dfm.csv')  # local file in server/
    except Exception as e:
        logger.error(f"Failed to load Male Diabetes Dataset: {e}")
        return

    Diabetes_DS_male = pd.get_dummies(Diabetes_DS_male, drop_first=True)
    X_male = Diabetes_DS_male.drop(['Diabetes'], axis=1)
    y_male = Diabetes_DS_male['Diabetes']
    X_male_columns = X_male.columns 

    poly_male = PolynomialFeatures(degree=2)
    X_poly = poly_male.fit_transform(X_male)
    scaler_male = StandardScaler().fit(X_poly)
    X_poly_scaled = scaler_male.transform(X_poly)
    X_train_male, X_val_male, y_train_male, y_val_male = tts(X_poly_scaled, y_male, test_size=0.2, random_state=0)
    lr_male = LogisticRegression()
    lr_male.fit(X_train_male, y_train_male)
    joblib.dump({"poly": poly_male, "scaler": scaler_male, "model": lr_male, "columns": X_male_columns}, _MALE_BUNDLE)
    logger.info("Male diabetes model trained and saved.")

def predict_male_diabetes(input_data):
    if lr_male is None:
        preprocess_male_diabetes()

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=X_male_columns, fill_value=0)
    X_poly_input = poly_male.transform(input_df)
    X_poly_scaled_input = scaler_male.transform(X_poly_input)
    
    probabilities = lr_male.predict_proba(X_poly_scaled_input)
    probability_of_positive_class = probabilities[0][1]
    
    return int(lr_male.predict(X_poly_scaled_input)[0]), probability_of_positive_class
