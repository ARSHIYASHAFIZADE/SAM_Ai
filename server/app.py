from flask import Flask, request, jsonify, session
from flask_bcrypt import Bcrypt
from flask_cors import CORS, cross_origin
from flask_session import Session
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from models import db, User
from config import ApplicationConfig
import pandas as pd
import numpy as np
import logging
import os
import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from sqlalchemy.orm.exc import NoResultFound
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app, supports_credentials=True )
app.config.from_object(ApplicationConfig)
bcrypt = Bcrypt(app)
server_Session = Session(app)
db.init_app(app)
@cross_origin
@app.route('/@me')
def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error":"unauthorized"}), 401
    user = User.query.filter_by(id=user_id).first()
    return jsonify({
        "id":user.id,
        "email":user.email
    })
@cross_origin
@app.route("/register", methods=["POST"])
def register_user():
    email = request.json['email']
    password = request.json['password']
    user_exist = User.query.filter_by(email=email).first() is not None
    if user_exist:
        return jsonify({"error":"user already exist"}), 409
    hashed_password = bcrypt.generate_password_hash(password)
    new_user = User(email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({
        "id":new_user.id,
        "email":new_user.email,
    })
@cross_origin
@app.route('/login', methods=['POST'])
def login_user():
    email = request.json['email']
    password = request.json['password']
    user = User.query.filter_by(email=email).first() 
    if user is None:
        return jsonify({"error":"unauthorized"}), 401
    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error":"unauthorized"}), 401  
    session["user_id"] = user.id
    return jsonify({
        "id":user.id,
        "email":user.email,
    })
with app.app_context():
    db.create_all()
# Load and preprocess dataset for female diabetes detection
def preprocess_female_diabetes():
    global Diabetes_DS, transformer, scaler, gbc, model, upper, feature_names
    
    # Get the path from an environment variable

    # Read the CSV file
    Diabetes_DS = pd.read_csv('dfw.csv')
    print("Current working directory:", Diabetes_DS)
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
    feature_names = X.columns.tolist()

    # Scale the features using RobustScaler
    transformer = RobustScaler().fit(X)
    X_scaled = transformer.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = tts(X_scaled, y, test_size=0.2, random_state=0)

    # Standardize the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train a Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(learning_rate=0.1, loss='exponential', n_estimators=150)
    gbc.fit(X_train_scaled, y_train)

    # Train a Logistic Regression model in a pipeline
    model = LogisticRegression(max_iter=1000)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train_scaled, y_train)

preprocess_female_diabetes()

# Prediction endpoint for female diabetes detection
# Prediction endpoint for female diabetes detection
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get('data')
        if not input_data:
            return jsonify({'error': 'No data provided'}), 400  
        logger.info(f"Received input data: {input_data}")  
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])  
        
        # Apply preprocessing steps to the input data
        input_df['NewBMI'] = pd.cut(input_df['BMI'], bins=[-np.inf, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf], labels=["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"])
        input_df['NewInsulinScore'] = input_df.apply(set_insulin, axis=1)
        input_df["NewGlucose"] = pd.cut(input_df["Glucose"], bins=[-np.inf, 70, 99, 126, np.inf], labels=["Low", "Normal", "Overweight", "High"])  

        # Generate dummies for categorical features
        input_df = pd.get_dummies(input_df, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)  
        
        # Reindex to match expected columns, ensuring correct order and filling missing columns with zeros
        input_df = input_df.reindex(columns=feature_names, fill_value=0)  
        
        # Scale the input data
        input_df_scaled = transformer.transform(input_df)  
        
        # Log the shapes of the data frames
        logger.info(f"Scaled input data shape: {input_df_scaled.shape}")  
        
        # Make prediction (0 or 1)
        prediction = gbc.predict(input_df_scaled)  # 0 or 1 (class label)
        probability = model.predict_proba(input_data)
        # Log the prediction result
        logger.info(f"Prediction result: {int(prediction[0])}") 
        
        prob = probability[0][1] * 100  # Convert to percentage
    
        return jsonify({
            'prediction': prediction,
            'probability': round(prob, 2)  # Round for cleaner output
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Load and preprocess dataset for male diabetes detection
def preprocess_male_diabetes():
    global Diabetes_DS_male, poly, scaler_male, lr, X_male
    Diabetes_DS_male = pd.read_csv('dfm.csv')
    Diabetes_DS_male = pd.get_dummies(Diabetes_DS_male, drop_first=True)
    X_male = Diabetes_DS_male.drop(['Diabetes'], axis=1)
    y_male = Diabetes_DS_male['Diabetes']
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_male)
    scaler_male = StandardScaler().fit(X_poly)
    X_poly_scaled = scaler_male.transform(X_poly)
    X_train_male, X_val_male, y_train_male, y_val_male = tts(X_poly_scaled, y_male, test_size=0.2, random_state=0)
    lr = LogisticRegression()
    lr.fit(X_train_male, y_train_male)
preprocess_male_diabetes()
# Prediction endpoint for male diabetes detection
@app.route('/predict_male', methods=['POST'])
def predict_male():
    try:
        input_data = request.json.get('data')
        if not input_data:
            return jsonify({'error': 'No data provided'}), 400
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=X_male.columns, fill_value=0)
        X_poly_input = poly.transform(input_df)
        X_poly_scaled_input = scaler_male.transform(X_poly_input)
        # Get the probability for the positive class (class 1)
        probabilities = lr.predict_proba(X_poly_scaled_input)
        probability_of_positive_class = probabilities[0][1]
        return jsonify({
            'prediction': int(lr.predict(X_poly_scaled_input)[0]),
            'probability': probability_of_positive_class
        }) 
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500
# Global variables for the preprocessor and model
preprocessor_transformer_heart = None
model_trained_heart = None
proba_have_heart_disease = None
def Heart_Disease_Detection(input_data_):
    global proba_have_heart_disease
    # Load the dataset
    data = pd.read_csv(('diseaseheart/heart_disease_data.csv'))
    print(data.columns)
    # Ensure the dataset includes all 14 fields
    expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    if not all(col in data.columns for col in expected_columns):
        raise ValueError('Dataset does not have the expected columns.')
    # Identify categorical and numerical features
    data_cat = data.select_dtypes("object")
    data_num = data.select_dtypes(["int64", "float64"])
    # One-hot encoding for categorical variables
    lbl = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    data_cat_ohe = lbl.fit_transform(data_cat)
    columns = lbl.get_feature_names_out()
    data_cat_ohe_df = pd.DataFrame(data_cat_ohe, columns=columns)
    # Combine numerical and categorical data
    data_final = pd.concat([data_num.reset_index(drop=True), data_cat_ohe_df.reset_index(drop=True)], axis=1)
    # Prepare features and target variable
    X = data_final.drop('target', axis=1)  # Use 'target' instead of 'HeartDisease'
    y = data_final['target']  # Use 'target' instead of 'HeartDisease'
    # Define numerical and categorical feature names
    num_features = X.select_dtypes(["int64", "float64"]).columns.tolist()
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
    global preprocessor_transformer_heart
    preprocessor_transformer_heart = preprocessor
    global model_trained_heart
    model_trained_heart = model
    # Prepare input data for prediction
    input_data = pd.DataFrame([input_data_])
    input_data_num = input_data.select_dtypes(["int64", "float64"])
    input_data_cat = input_data.select_dtypes("object")
    # One-hot encode categorical input data
    if not input_data_cat.empty:
        input_data_cat_ohe = lbl.transform(input_data_cat)
        input_data_cat_ohe_df = pd.DataFrame(input_data_cat_ohe, columns=columns)
    else:
        input_data_cat_ohe_df = pd.DataFrame(columns=columns)
    # Combine numerical and categorical data
    input_data_final = pd.concat([input_data_num.reset_index(drop=True), input_data_cat_ohe_df.reset_index(drop=True)], axis=1)
    # Ensure all required features are present
    for col in X.columns:
        if col not in input_data_final.columns:
            input_data_final[col] = 0
    input_data_final = input_data_final[X.columns]
    # Apply saved transformers to the input data
    input_data_preprocessed = preprocessor_transformer_heart.transform(input_data_final)
    # Prediction & probability of heart disease
    prediction = model_trained_heart.predict(input_data_preprocessed)
    probabilities = model_trained_heart.predict_proba(input_data_preprocessed)
    proba_have_heart_disease = f'{probabilities[0][1] * 100:.2f}'
    return prediction[0]
@app.route('/detect_heart', methods=['POST'])
def detect_heart():
    try:
        data = request.get_json()
        print('Received data:', data)
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        if not all(key in data.get('data', {}) for key in required_fields):
            raise ValueError('Missing required data fields')
        input_data = [data['data'][field] for field in required_fields]
        result = Heart_Disease_Detection(dict(zip(required_fields, input_data)))    
        response = {
            'prediction': 'Heart disease' if result == 1 else 'No heart disease',
            'probability': proba_have_heart_disease
        }    
        return jsonify(response)
    except Exception as e:
        print(f'Error: {str(e)}')
        return jsonify({'error': str(e)}), 500
# Global variables to store trained model and scaler
gbm_model_liver = None
scaler_liver = None
def train_liver_model():
    global gbm_model_liver, scaler_liver
    # Load the dataset
    Liver_DS = pd.read_csv('liver/Liver_disease_data.csv')
    # Prepare features and target variable
    X = Liver_DS.drop(columns='Diagnosis', axis=1)
    Y = Liver_DS['Diagnosis']   
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split the data into training and validation sets
    x_train, x_va, y_train, y_va = tts(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
    # Train the Gradient Boosting model
    gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.121, random_state=103)
    gbm_model.fit(x_train, y_train)
    # Save the trained model and scaler for later use
    gbm_model_liver = gbm_model
    scaler_liver = scaler
def Healthy_Liver_Detection(input_data_):
    global gbm_model_liver, scaler_liver
    if gbm_model_liver is None or scaler_liver is None:
        # If model or scaler are not loaded, train them
        train_liver_model()
    # Prepare the input data for prediction
    input_data_as_numpy_array = np.asarray(input_data_)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    standardized_input_data = scaler_liver.transform(input_data_reshape)
    # Make prediction
    probabilities = gbm_model_liver.predict_proba(standardized_input_data)[0]  # Get both class probabilities
    prediction = np.argmax(probabilities)  # Get the class with the highest probability
    # Format the probability
    if prediction == 0:
        probability_healthy_liver = f"{probabilities[0] * 100:.2f}%"
    else:
        probability_healthy_liver = f"{probabilities[1] * 100:.2f}%"
    # Convert prediction to a standard Python int
    prediction = int(prediction)
    return prediction, probability_healthy_liver
@app.route('/detect_liver', methods=['POST'])
def detect_liver():
    input_data = request.json['input_data']  # Assuming the input data is passed as JSON
    prediction, probability_healthy_liver = Healthy_Liver_Detection(input_data)
    return jsonify({
        'prediction': prediction,  # 0 for healthy, 1 for liver problems
        'probability_healthy_liver': probability_healthy_liver  # Formatted probability with %
    })
# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'a6833351@gmail.com'
app.config['MAIL_PASSWORD'] = 'fxfm lfzl gwme oohz'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_DEFAULT_SENDER'] = 'a6833351@gmail.com'
mail = Mail(app)

# Global variable to store probability of having breast cancer
proba_have_breast_cancer = None

# Load and prepare data
Breast_Cancer_DS = load_breast_cancer()
df = pd.DataFrame(Breast_Cancer_DS.data, columns=Breast_Cancer_DS.feature_names)
df['diagnosis'] = Breast_Cancer_DS.target
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']
x_train, x_va, y_train, y_va = tts(X, y, test_size=0.3, random_state=3)
scaler = StandardScaler()
x_scaled_tr = scaler.fit_transform(x_train)
LR = LogisticRegression(penalty='l2', max_iter=100)
LR.fit(x_scaled_tr, y_train)

def Breast_Cancer_Detection(input_data_):
    global proba_have_breast_cancer
    # Convert input data to numpy array
    input_data_as_numpy_array = np.array([
        input_data_['mean_radius'],
        input_data_['mean_texture'],
        input_data_['mean_perimeter'],
        input_data_['mean_area'],
        input_data_['mean_smoothness'],
        input_data_['mean_compactness'],
        input_data_['mean_concavity'],
        input_data_['mean_concave_points'],
        input_data_['mean_symmetry'],
        input_data_['mean_fractal_dimension'],
        input_data_['radius_error'],
        input_data_['texture_error'],
        input_data_['perimeter_error'],
        input_data_['area_error'],
        input_data_['smoothness_error'],
        input_data_['compactness_error'],
        input_data_['concavity_error'],
        input_data_['concave_points_error'],
        input_data_['symmetry_error'],
        input_data_['fractal_dimension_error'],
        input_data_['worst_radius'],
        input_data_['worst_texture'],
        input_data_['worst_perimeter'],
        input_data_['worst_area'],
        input_data_['worst_smoothness'],
        input_data_['worst_compactness'],
        input_data_['worst_concavity'],
        input_data_['worst_concave_points'],
        input_data_['worst_symmetry'],
        input_data_['worst_fractal_dimension']
    ], dtype=float)
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Check if input_data has the correct number of features
    if input_data_reshaped.shape[1] != X.shape[1]:
        raise ValueError(f"Expected {X.shape[1]} features, but got {input_data_reshaped.shape[1]} features")
    
    std_data = scaler.transform(input_data_reshaped)
    prediction = LR.predict(std_data)
    probabilities = LR.predict_proba(std_data)
    
    # Determine probability of malignant cancer
    proba_have_breast_cancer = f'{probabilities[0][1] * 100:.2f}'
    
    # Print debugging info
    print(f"Prediction: {prediction[0]}, Probability: {proba_have_breast_cancer}%")
    
    # Return both prediction and probability
    return int(prediction[0]), float(proba_have_breast_cancer)

def plot_charts(input_data, input_names):
    # Bar Chart
    plt.figure(figsize=(10, 8))
    sns.barplot(x=input_names, y=input_data, palette="YlGnBu")
    plt.title('Input Data')
    plt.xlabel('Variables')
    plt.ylabel('Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('input_data_plot_Breast_Cancer.png')
    plt.close()

    # Radar Chart
    stats = np.array(input_data)
    labels = np.array(input_names)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, color='blue', linewidth=2, linestyle='solid', label='Individual')
    ax.fill(angles, stats, color='green', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    plt.title('Breast Cancer Data Radar Chart', size=15, color='blue', y=1.1)
    plt.savefig('radar_chart_Breast_Cancer.png')
    plt.close()

def create_pdf_report(filename, result, proba, input_data, input_names):
    if len(input_data) != len(input_names):
        raise ValueError("Length of input_data and input_names must be the same.")
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    header_style = ParagraphStyle('HeaderStyle', parent=styles['Heading2'], spaceAfter=14)

    # Title and Introduction
    title = Paragraph("BREAST CANCER DETECTION REPORT", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    Breast_Cancer_Detection_title = Paragraph("Breast Cancer Detection", header_style)
    elements.append(Breast_Cancer_Detection_title)
    elements.append(Spacer(1, 5))

    # Data Table
    input_data_table_data = [('FEATURES', 'VALUES')] + [(input_names[i], input_data[i]) for i in range(len(input_data))]
    input_data_table_data.append(('Result', result))
    input_data_table_data.append(('Probability', proba))
    prob_color = colors.red if float(proba) > 80 else (colors.orange if float(proba) > 50 else colors.green)
    input_data_table = Table(input_data_table_data, colWidths=[160, 300])
    input_data_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (1, -1), (1, -1), prob_color),
        ('BACKGROUND', (1, -2), (1, -2), colors.lightgrey),
        ('TEXTCOLOR', (1, -1), (1, -1), colors.white),
        ('TEXTCOLOR', (1, -2), (1, -2), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(input_data_table)
    elements.append(Spacer(1, 12))
    elements.append(PageBreak())

    # Generate Charts
    plot_charts(input_data, input_names)
    input_data_plot = Image('input_data_plot_Breast_Cancer.png', width=400, height=200)
    radar_chart_plot = Image('radar_chart_Breast_Cancer.png', width=400, height=400)
    elements.append(input_data_plot)
    elements.append(Spacer(1, 12))
    elements.append(radar_chart_plot)
    elements.append(Spacer(1, 12))

    # Build PDF
    doc.build(elements)

def send_email(to_email, subj, body_, attachment_path):
    # Email sending logic
    sender = 'a6833351@gmail.com'
    password = 'fxfm lfzl gwme oohz'
    receiver = to_email
    subject = subj
    body = body_
    attach_path = attachment_path

    # Create message of email
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach PDF
    with open(attach_path, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename={attachment_path}',
        )
        msg.attach(part)

    # Send email
    try:
        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
            server.starttls()
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            server.send_message(msg)
    except Exception as e:
        print(f"Error sending email: {e}")

@app.route('/detect_breast_cancer', methods=['POST'])
def detect_breast_cancer():
    try:
        # Fetch the current user's ID from the session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Unauthorized: No user logged in"}), 401
        
        # Fetch the current user's email from the database
        try:
            user = User.query.filter_by(id=user_id).one()
            recipient_email = user.email
        except NoResultFound:
            return jsonify({"error": "User not found"}), 404
        
        # Process breast cancer detection
        input_data_ = request.json
        prediction, proba = Breast_Cancer_Detection(input_data_)
        
        # Ensure the result is a standard Python int
        result = int(prediction)
        
        # Prepare data for PDF
        input_data_values = [input_data_[key] for key in input_data_.keys()]
        input_names = list(input_data_.keys())
        pdf_filename = 'Breast_Cancer_Detection_Report.pdf'
        create_pdf_report(pdf_filename, result, proba, input_data_values, input_names)
        
        # Send email with PDF
        subject = 'Breast Cancer Detection Report'
        body = 'Please find attached the breast cancer detection report.'
        send_email(recipient_email, subject, body, pdf_filename)
        
        # Adjusted response format to match React component expectations
        response = {
            "prediction": result,
            "probability_breast_cancer": float(proba)  # Ensure probability is a float
        }
        return jsonify(response), 200
    except Exception as e:
        print(f"Exception occurred: {e}")
        return jsonify({"error": str(e)}), 400

if __name__=="__main__":
    app.run()