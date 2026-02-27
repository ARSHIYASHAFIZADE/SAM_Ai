# SAM AI — Clinical Diagnostic Suite

![Next.js](https://img.shields.io/badge/Next.js-15.0-000000?logo=nextdotjs&logoColor=white)
![React](https://img.shields.io/badge/React-19.0-61DAFB?logo=react&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Mistral--7B-FFD21E?logo=huggingface&logoColor=black)

**SAM AI** is a premium, full-stack health-tech application that leverages machine learning models to provide early-stage risk assessments for multiple critical diseases. It features a professional clinical interface, a built-in AI medical assistant, and sophisticated data analysis.

## Features

- **Premium Clinical UI**: Modern, fully responsive, glassmorphic design built with Tailwind CSS v4 and Framer Motion.
- **AI Medical Assistant**: Integrated chat widget powered by Hugging Face Inference API (Mistral-7B-Instruct) for general health Q&A.
- **Secure Architecture**: Server-side API routing, Redis-backed session management, and robust input validation.
- **Detailed PDF Reports**: Automated generation and emailing of comprehensive diagnostic results (via SendGrid/Flask-Mail).

---

## Diagnostic Modules

| Specialty | Targeted Assessment | Key Biomarkers / Features |
|-----------|---------------------|---------------------------|
| **Cardiology** | ❤️ Heart Disease Risk | 13 vitals (e.g., resting ECG, cholesterol, max heart rate) |
| **Oncology** | 🔬 Breast Cancer Screening | 30 FNA cell nucleus measurements (e.g., radius, texture, area) |
| **Endocrinology** | 🩸 Female Diabetes Risk | 8 PIMA metabolic endpoints (e.g., glucose, BMI, insulin) |
| **Endocrinology** | 💉 Male Diabetes Risk | 16 clinical symptom indicators (e.g., polyuria, polydipsia) |
| **Hepatology** | 🫁 Liver Health | 10 hepatic enzyme panel measurements (e.g., ALT, AST, bilirubin) |

*Results feature detailed probability scoring, dynamic risk-level badging, confidence bars, and (for oncology) dynamic radar/bar chart visualizations.*

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js (App Router), React, TypeScript, Tailwind CSS v4, Framer Motion, React Three Fiber |
| **Backend** | Python, Flask, Redis (Sessions), Flask-Limiter, Pydantic (data validation) |
| **Database** | PostgreSQL (Production) / SQLite (Dev) |
| **Machine Learning** | scikit-learn, joblib, pandas |
| **AI Integration** | Hugging Face Serverless API (`Mistral-7B-Instruct-v0.2`) |
| **Deployment** | Vercel (Frontend) + Railway (Backend) |

---

## Run It Locally

**Prerequisites:** Node.js 18+ and Python 3.9+

```bash
# Clone the repository
git clone https://github.com/ARSHIYASHAFIZADE/SAM_Ai.git
cd SAM_Ai
```

### 1. Backend Setup
```bash
cd server
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```
*Configure environment variables in `server/.env` first (see below), then run:*
```bash
python app.py                # Starts on http://localhost:5000
```

### 2. Frontend Setup
```bash
cd sam-next
npm install
```
*Create a `.env.local` file containing:*
```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:5000
HF_TOKEN=your_huggingface_read_token
```
*Then start the development server:*
```bash
npm run dev                  # Starts on http://localhost:3000
```

---

## ⚠️ Disclaimer
This is a personal project built for educational, learning, and demonstration purposes. **The predictions are based on machine learning models trained on public datasets and are strictly NOT medical advice.** Always consult a certified medical professional for real health concerns.

## License
MIT — See [LICENSE](https://opensource.org/licenses/MIT).
