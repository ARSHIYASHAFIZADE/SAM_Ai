#!/usr/bin/env python3
"""
Pre-training script for SAM_Ai ML models.
Generates pickle files for all 5 models to avoid runtime training on Render.

Usage:
    python train_models.py

This creates saved_models/*.pkl files that will be loaded at startup instead of training on each deploy.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_all_models():
    """Train and save all ML models."""
    logger.info("Starting model pre-training...")

    try:
        from services.diabetes_service import preprocess_female_diabetes, preprocess_male_diabetes
        from services.heart_service import train_heart_model
        from services.liver_service import train_liver_model
        from services.cancer_service import train_cancer_model

        logger.info("Training Female Diabetes Model...")
        preprocess_female_diabetes()

        logger.info("Training Male Diabetes Model...")
        preprocess_male_diabetes()

        logger.info("Training Heart Disease Model...")
        train_heart_model()

        logger.info("Training Liver Disease Model...")
        train_liver_model()

        logger.info("Training Cancer Detection Model...")
        train_cancer_model()

        logger.info("✓ All models trained and saved successfully!")
        logger.info("Models are saved in: ./saved_models/")
        logger.info("On deployment, these will load instantly instead of training at startup.")
        return True

    except Exception as e:
        logger.error(f"✗ Model training failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = train_all_models()
    sys.exit(0 if success else 1)
