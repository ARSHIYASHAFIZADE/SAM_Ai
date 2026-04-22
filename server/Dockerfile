FROM python:3.11-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --prefer-binary -r requirements.txt gunicorn

COPY . .

# Pre-train ML models during build to avoid OOM at runtime
# This creates saved_models/*.pkl files that will be loaded instantly on startup
RUN echo "Pre-training ML models..." && \
    python -c "from services.diabetes_service import preprocess_female_diabetes, preprocess_male_diabetes; \
               from services.heart_service import train_heart_model; \
               from services.liver_service import train_liver_model; \
               from services.cancer_service import train_cancer_model; \
               preprocess_female_diabetes(); \
               preprocess_male_diabetes(); \
               train_heart_model(); \
               train_liver_model(); \
               train_cancer_model(); \
               print('✓ All models pre-trained successfully')" || \
    echo "⚠ Model pre-training skipped (models may not be available)"

EXPOSE 8000

CMD ["gunicorn", "--workers=1", "--worker-class=sync", "--bind", "0.0.0.0:8000", "--timeout", "120", "app:app"]
