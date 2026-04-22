import matplotlib
matplotlib.use("Agg")

import io
import base64
import logging
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

logger = logging.getLogger(__name__)

_MODELS_DIR = Path("saved_models")
_CANCER_BUNDLE = _MODELS_DIR / "cancer_bundle.pkl"

scaler_breast_cancer = None
model_breast_cancer = None
X_shape_1 = 30


def train_cancer_model():
    global scaler_breast_cancer, model_breast_cancer, X_shape_1
    _MODELS_DIR.mkdir(exist_ok=True)

    if _CANCER_BUNDLE.exists():
        b = joblib.load(_CANCER_BUNDLE)
        scaler_breast_cancer = b["scaler"]
        model_breast_cancer = b["model"]
        X_shape_1 = b["X_shape_1"]
        logger.info("Cancer model loaded from disk.")
        return

    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['diagnosis'] = dataset.target

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_shape_1 = X.shape[1]

    scaler_breast_cancer = StandardScaler()
    X_scaled = scaler_breast_cancer.fit_transform(X)

    model_breast_cancer = LogisticRegression(max_iter=10000)
    model_breast_cancer.fit(X_scaled, y)

    joblib.dump(
        {"scaler": scaler_breast_cancer, "model": model_breast_cancer, "X_shape_1": X_shape_1},
        _CANCER_BUNDLE,
    )
    logger.info("Cancer model trained and saved.")


def predict_breast_cancer(input_data: dict):
    if model_breast_cancer is None or scaler_breast_cancer is None:
        train_cancer_model()

    values = np.array(list(input_data.values()), dtype=float).reshape(1, -1)

    if values.shape[1] != X_shape_1:
        raise ValueError(f"Expected {X_shape_1} features, got {values.shape[1]}")

    scaled = scaler_breast_cancer.transform(values)
    prediction = model_breast_cancer.predict(scaled)
    probabilities = model_breast_cancer.predict_proba(scaled)

    prob_malignant = round(probabilities[0][0] * 100, 2)
    return int(prediction[0]), float(prob_malignant)


def get_breast_cancer_visualizations(input_data: dict, input_names: list) -> dict:
    input_values = [input_data[name] for name in input_names]
    charts = {}

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=input_names, y=input_values, palette="YlGnBu", ax=ax)
    ax.set_title('Feature Values')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    charts['bar_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # Radar chart
    stats = np.array(input_values, dtype=float)
    num_vars = len(input_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    stats_closed = np.concatenate((stats, [stats[0]]))
    angles_closed = angles + angles[:1]

    fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax2.plot(angles_closed, stats_closed, color='#14b8a6', linewidth=2)
    ax2.fill(angles_closed, stats_closed, color='#14b8a6', alpha=0.2)
    ax2.set_yticklabels([])
    ax2.set_xticks(angles)
    ax2.set_xticklabels(input_names, fontsize=7)
    ax2.set_title('Feature Distribution', size=13, y=1.08)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    charts['radar_chart'] = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close(fig2)

    return charts
