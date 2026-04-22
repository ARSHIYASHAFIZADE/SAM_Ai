<div align="center">

# SAM AI — Clinical Diagnostic Suite

**Early disease detection powered by machine learning and AI-assisted chat**

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## Demo

<div align="center">

![SAM AI demo — landing page, login, heart disease result, breast cancer radar chart, AI chat, dashboard](docs/demo.gif)

<sub>Landing · Login · Heart Disease · Breast Cancer radar chart · AI Chat · Dashboard &nbsp;|&nbsp; <a href="docs/demo.mp4">Full demo MP4</a></sub>

</div>

---

## What It Does

SAM AI lets users fill in clinical biomarkers and receive an instant ML-powered risk assessment — complete with a confidence score, risk badge, clinical recommendations, and a downloadable PDF report. A built-in AI chat widget (powered by Groq / Llama 3.3-70B) answers general health questions.

---

## Diagnostic Modules

| Specialty | Assessment | Key Inputs |
|-----------|-----------|-----------|
| **Cardiology** | Heart Disease Risk | 13 vitals — resting ECG, cholesterol, max heart rate, ST depression |
| **Oncology** | Breast Cancer Screening | 30 FNA cell-nucleus measurements — radius, texture, area, concavity |
| **Endocrinology** | Female Diabetes Risk | 8 PIMA endpoints — glucose, BMI, insulin, pregnancies |
| **Endocrinology** | Male Diabetes Risk | 16 symptom indicators — polyuria, polydipsia, weight loss |
| **Hepatology** | Liver Health | 10 hepatic enzyme values — ALT, AST, bilirubin, albumin |

Each result includes:
- Probability score and risk level (Low / Moderate / High)
- Confidence bar with model version
- Clinical recommendations panel
- PDF report download (generated client-side with jsPDF)
- Radar / bar chart for breast cancer oncology results

---

## Tech Stack

### Frontend

| | |
|---|---|
| Framework | Next.js 15 (App Router) + React 19 |
| Language | TypeScript 5.x |
| Styling | Tailwind CSS v4 + Framer Motion |

### Backend

| | |
|---|---|
| Framework | Flask 3.x + Flask-Session + Flask-Limiter |
| Language | Python 3.9+ |
| Database | SQLite (dev) — swap `DATABASE_URL` for PostgreSQL in production |
| Auth | Flask-Login + bcrypt + server-side sessions (filesystem) |
| ML | scikit-learn, joblib, pandas, NumPy |

### AI Chat

Powered by **Groq** inference (Llama 3.3-70B). Set `GROQ_API_KEY` in `sam-next/.env.local` to enable it. Free tier available at [console.groq.com](https://console.groq.com).

---

## Getting Started

**Prerequisites:** Node.js 18+ and Python 3.9+

```bash
git clone https://github.com/ARSHIYASHAFIZADE/SAM_Ai.git
cd SAM_Ai
```

### 1. Backend

```bash
cd server
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `server/.env`:

```env
SECRET_KEY=change-me-in-production
DATABASE_URL=sqlite:///./db.sqlite
SESSION_TYPE=filesystem
SESSION_COOKIE_SECURE=false
GROQ_API_KEY=                   # optional — only needed for the chat route
```

Start the backend:

```bash
python app.py                   # http://localhost:5000
```

### 2. Frontend

```bash
cd sam-next
npm install
```

Create `sam-next/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:3000/api-backend
BACKEND_URL=http://localhost:5000
GROQ_API_KEY=your_groq_key_here
```

> The frontend proxies all `/api-backend/*` requests to the Flask backend via a Next.js rewrite. This keeps auth cookies on the same origin and avoids CORS issues.

Start the frontend:

```bash
npm run dev                     # http://localhost:3000
```

---

## Project Structure

```
SAM_Ai/
├── sam-next/                    # Next.js 15 frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # Landing page
│   │   │   ├── register/        # Registration
│   │   │   ├── login/           # Login
│   │   │   ├── dashboard/       # Assessment history
│   │   │   ├── predict/
│   │   │   │   ├── heart/       # Heart disease form + result
│   │   │   │   ├── diabetes-female/
│   │   │   │   ├── diabetes-male/
│   │   │   │   ├── liver/
│   │   │   │   └── breast-cancer/
│   │   │   └── api/chat/        # Groq / Llama chat route
│   │   └── components/
│   │       ├── Navbar.tsx
│   │       ├── AuthProvider.tsx  # Session context + RequireAuth HOC
│   │       ├── ResultCard.tsx    # Shared result display
│   │       ├── RecommendationsPanel.tsx
│   │       ├── ChatWidget.tsx    # Floating AI chat button + panel
│   │       └── MedicalIcons.tsx
│   └── next.config.js           # Rewrites /api-backend/* → Flask
│
├── server/                      # Flask backend
│   ├── app.py                   # App factory, routes, CORS
│   ├── config.py                # ApplicationConfig (reads .env)
│   ├── models.py                # SQLAlchemy User model
│   ├── schemas.py               # Pydantic request schemas
│   └── services/
│       ├── heart_service.py
│       ├── female_diabetes_service.py
│       ├── male_diabetes_service.py
│       ├── liver_service.py
│       └── cancer_service.py    # Breast cancer + chart generation
│
└── docs/
    └── demo.mp4
```

---

## API Reference

All routes are prefixed with `/` on the Flask backend (port 5000), accessed from the frontend via `/api-backend/`.

### Auth

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/register` | Create account — `{ name, email, password }` |
| `POST` | `/login` | Sign in — `{ email, password }` |
| `POST` | `/logout` | Clear session |
| `GET` | `/@me` | Return current user or 401 |

### Predictions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/detect_heart` | Heart disease — 13 numeric + categorical fields |
| `POST` | `/predict` | Female diabetes — 8 numeric fields |
| `POST` | `/predict_male` | Male diabetes — Age (float) + 15 Yes/No/Male/Female fields |
| `POST` | `/detect_liver` | Liver health — `{ input_data: [10 floats] }` |
| `POST` | `/detect_breast_cancer` | Breast cancer — 30 numeric measurements |

Each prediction returns `{ prediction, probability, risk_level, model_version }`. Breast cancer also returns `bar_chart` and `radar_chart` as base64-encoded PNG strings.

### AI Chat

The chat widget calls the Next.js API route at `/api/chat` (not the Flask backend), which proxies to Groq.

---

## Environment Variables

### Backend (`server/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | — | **Required.** Flask session signing key |
| `DATABASE_URL` | `sqlite:///./db.sqlite` | SQLAlchemy database URL |
| `SESSION_TYPE` | `filesystem` | Flask-Session backend |
| `SESSION_COOKIE_SECURE` | `true` | Set `false` for HTTP localhost |
| `GROQ_API_KEY` | — | Optional — only needed if calling Groq from the backend |

### Frontend (`sam-next/.env.local`)

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_BASE_URL` | Must be `http://localhost:3000/api-backend` (same-origin proxy) |
| `BACKEND_URL` | Where Next.js rewrites proxy to — `http://localhost:5000` |
| `GROQ_API_KEY` | Groq API key for the `/api/chat` Next.js route |

---

## Disclaimer

SAM AI is built for educational and demonstration purposes. Predictions are produced by ML models trained on public datasets and are **not** a substitute for professional medical advice. Always consult a qualified healthcare provider for real health concerns.

---

## License

MIT
