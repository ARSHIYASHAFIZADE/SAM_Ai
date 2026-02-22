import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import logging
import smtplib
import io
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

logger = logging.getLogger(__name__)

scaler_breast_cancer = None
model_breast_cancer = None
X_shape_1 = 30 # Default

def train_cancer_model():
    global scaler_breast_cancer, model_breast_cancer, X_shape_1
    
    Breast_Cancer_DS = load_breast_cancer()
    df = pd.DataFrame(Breast_Cancer_DS.data, columns=Breast_Cancer_DS.feature_names)
    df['diagnosis'] = Breast_Cancer_DS.target
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    X_shape_1 = X.shape[1]
    
    x_train, x_va, y_train, y_va = tts(X, y, test_size=0.3, random_state=3)
    
    scaler_breast_cancer = StandardScaler()
    x_scaled_tr = scaler_breast_cancer.fit_transform(x_train)
    
    LR = LogisticRegression(penalty='l2', max_iter=100)
    LR.fit(x_scaled_tr, y_train)
    
    model_breast_cancer = LR

def predict_breast_cancer(input_data_):
    if model_breast_cancer is None:
        train_cancer_model()

    # Convert input data to numpy array in correct order
    # Assuming input_data_ is a dict
    input_data_as_numpy_array = np.array([
        input_data_['mean_radius'], input_data_['mean_texture'], input_data_['mean_perimeter'],
        input_data_['mean_area'], input_data_['mean_smoothness'], input_data_['mean_compactness'],
        input_data_['mean_concavity'], input_data_['mean_concave_points'], input_data_['mean_symmetry'],
        input_data_['mean_fractal_dimension'], input_data_['radius_error'], input_data_['texture_error'],
        input_data_['perimeter_error'], input_data_['area_error'], input_data_['smoothness_error'],
        input_data_['compactness_error'], input_data_['concavity_error'], input_data_['concave_points_error'],
        input_data_['symmetry_error'], input_data_['fractal_dimension_error'], input_data_['worst_radius'],
        input_data_['worst_texture'], input_data_['worst_perimeter'], input_data_['worst_area'],
        input_data_['worst_smoothness'], input_data_['worst_compactness'], input_data_['worst_concavity'],
        input_data_['worst_concave_points'], input_data_['worst_symmetry'], input_data_['worst_fractal_dimension']
    ], dtype=float)
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    if input_data_reshaped.shape[1] != X_shape_1:
         raise ValueError(f"Expected {X_shape_1} features, but got {input_data_reshaped.shape[1]} features")

    std_data = scaler_breast_cancer.transform(input_data_reshaped)
    prediction = model_breast_cancer.predict(std_data)
    probabilities = model_breast_cancer.predict_proba(std_data)
    
    # Probability of Malignant (Class 0)
    prob_val = probabilities[0][0] * 100
    prob_str = f'{prob_val:.2f}'
    
    return int(prediction[0]), float(prob_str)

def get_breast_cancer_visualizations(input_data_, input_names_):
    # Convert input dict to values list if needed, but assuming input_data_ is dict
    # We need values in specific order for plotting? 
    # The plot_charts function took input_data as list and input_names as list.
    
    # helper to get values list
    input_values = [input_data_[name] for name in input_names_]
    
    return plot_charts(input_values, input_names_)

# ... (Include PDF and Email logic here as helper functions, moved from app.py) ...
# For brevity in this turn, I will assume PDF/Email logic is copied here or imported.
# Let's include the essential PDF parts.

def plot_charts(input_data, input_names):
    charts = {}
    
    # Bar Chart
    plt.figure(figsize=(10, 8))
    sns.barplot(x=input_names, y=input_data, palette="YlGnBu")
    plt.title('Input Data')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to buffer instead of file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    charts['bar_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Radar Chart
    stats = np.array(input_data)
    labels = np.array(input_names)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, color='blue', linewidth=2, linestyle='solid')
    ax.fill(angles, stats, color='green', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    plt.title('Breast Cancer Data Radar Chart', size=15, color='blue', y=1.1)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    charts['radar_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return charts

def create_pdf_report(filename, result, proba, input_data, input_names):
    if len(input_data) != len(input_names):
        raise ValueError("Length mismatch")
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    title = Paragraph("BREAST CANCER DETECTION REPORT", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    table_data = [('FEATURES', 'VALUES')] + [(input_names[i], input_data[i]) for i in range(len(input_data))]
    table_data.append(('Result', result))
    table_data.append(('Probability', proba))
    
    t = Table(table_data, colWidths=[160, 300])
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
    ]))
    elements.append(t)
    elements.append(PageBreak())
    
    plot_charts(input_data, input_names)
    elements.append(Image('input_data_plot_Breast_Cancer.png', width=400, height=200))
    elements.append(Spacer(1, 12))
    elements.append(Image('radar_chart_Breast_Cancer.png', width=400, height=400))
    
    doc.build(elements)

def send_email(to_email, subj, body_, attachment_path, mail_ext, app_config):
    msg = MIMEMultipart()
    msg['From'] = app_config['MAIL_DEFAULT_SENDER'] # Or from config
    msg['To'] = to_email
    msg['Subject'] = subj
    msg.attach(MIMEText(body_, 'plain'))
    
    with open(attachment_path, 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={attachment_path}')
        msg.attach(part)
        
    # Using flask-mail or direct SMTP? Original used smtplib manually inside Flask context but with app.config attrs.
    # Let's use the passed 'mail' extension if possible, or manual SMTP using passed config.
    # Original used `with smtplib.SMTP`.
    try:
        with smtplib.SMTP(app_config['MAIL_SERVER'], app_config['MAIL_PORT']) as server:
            server.starttls()
            server.login(app_config['MAIL_USERNAME'], app_config['MAIL_PASSWORD'])
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Email error: {e}")
