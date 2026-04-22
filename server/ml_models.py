"""ML production utilities: versioning, response builder, feature importance."""

from datetime import datetime, timezone

MODEL_VERSION = "1.0.0"

# Registry — populated in app.py after training
_models_ready: dict = {}


def set_model_ready(name: str):
    _models_ready[name] = True


def all_models_ready() -> bool:
    return all(_models_ready.get(k) for k in
               ("female_diabetes", "male_diabetes", "heart", "liver", "cancer"))


def risk_level(prob_pct: float) -> str:
    """prob_pct must be 0–100."""
    if prob_pct < 30:
        return "Low"
    if prob_pct < 60:
        return "Moderate"
    return "High"


def normalize_prob(p) -> float:
    """Normalize any probability representation to 0–100."""
    if isinstance(p, str):
        return float(p.replace('%', '').strip())
    v = float(p)
    return v * 100 if v <= 1.0 else v


def get_feature_importance(model, feature_names=None) -> dict | None:
    if hasattr(model, 'feature_importances_'):
        vals = model.feature_importances_.tolist()
    elif hasattr(model, 'coef_'):
        vals = [abs(x) for x in model.coef_[0].tolist()]
    else:
        return None
    if feature_names and len(feature_names) == len(vals):
        return {k: round(v, 6) for k, v in zip(feature_names, vals)}
    return None


def build_response(prediction: int, probability_raw, feature_importance=None) -> dict:
    prob_pct = normalize_prob(probability_raw)
    resp = {
        "prediction": prediction,
        "probability": round(prob_pct, 2),
        "risk_level": risk_level(prob_pct),
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if feature_importance:
        resp["feature_importance"] = feature_importance
    return resp
