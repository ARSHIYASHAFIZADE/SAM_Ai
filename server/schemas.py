"""
Request body validation schemas using Pydantic v2.
All /predict* routes validate incoming JSON against these models
before any business logic runs.
"""

from pydantic import BaseModel, field_validator
from typing import List


# ── Female Diabetes ──────────────────────────────────────────────────────────
class FemaleDiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


# ── Male Diabetes ─────────────────────────────────────────────────────────────
class MaleDiabetesInput(BaseModel):
    """
    Accepts any dict that matches the dfm.csv feature set.
    Keep it open (extra fields allowed) so frontend field additions don't break validation.
    """
    model_config = {"extra": "allow"}

    Age: float
    Gender: str
    Polyuria: str
    Polydipsia: str
    sudden_weight_loss: str
    weakness: str
    Polyphagia: str
    Genital_thrush: str
    visual_blurring: str
    Itching: str
    Irritability: str
    delayed_healing: str
    partial_paresis: str
    muscle_stiffness: str
    Alopecia: str
    Obesity: str


# ── Heart Disease ─────────────────────────────────────────────────────────────
class HeartDiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


# ── Liver Health ──────────────────────────────────────────────────────────────
class LiverInput(BaseModel):
    input_data: List[float]

    @field_validator("input_data")
    @classmethod
    def must_have_ten_features(cls, v: List[float]) -> List[float]:
        if len(v) != 10:
            raise ValueError(f"liver input_data must have exactly 10 values, got {len(v)}")
        return v


# ── Breast Cancer ─────────────────────────────────────────────────────────────
class BreastCancerInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float
