# SAM AI: Intelligent Medical Diagnostic Assistant

SAM AI is a diagnostic support platform that utilizes machine learning models to provide clinical-grade health insights. Designed for both healthcare efficiency and patient accessibility, the application offers high-precision analysis for chronic conditions, integrated report management, and AI-assisted health consultations.

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://sam-ai-7lwa.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🔬 Predictive Capabilities
The application provides specialized diagnostic modules for high-risk conditions, using trained models to evaluate risk factors and clinical data:

- **Cardiovascular Health**: Analysis of heart disease risk factors.
- **Oncology (Breast Cancer)**: High-accuracy detection based on clinical indicators.
- **Metabolic Health**: Specialized prediction models for Diabetes.
- **Hepatology**: Diagnostic insights for Liver disease.

## ✨ Core Features
- **Integrated Health Dashboard**: A centralized portal to review, manage, and download historical diagnostic reports.
- **Consultation Interface**: Direct access to health queries with responses powered by medical-context models.
- **Transparent Insights**: Actionable results designed to bridge the gap between complex data and patient understanding.

## 🛠 Technology Stack
- **Frontend**: React (Vite), TypeScript, Vanilla CSS
- **Backend**: Flask (Python), SQLAlchemy
- **Database**: SQLite / PostgreSQL
- **Deployment**: Render

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.9+)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ARSHIYASHAFIZADE/SAM_Ai.git
   cd SAM_Ai
   ```

2. **Frontend Setup**
   ```bash
   cd sam
   npm install
   ```

3. **Backend Setup**
   ```bash
   cd ../server
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running Locally

1. **Start the Backend** (From `/server`)
   ```bash
   python app.py
   ```

2. **Start the Frontend** (From `/sam`)
   ```bash
   npm run dev
   ```

---

## 📸 Interface Preview
![Dashboard Preview](https://github.com/user-attachments/assets/d23467f5-2dbe-4105-8a9c-67cc254a97e9)

---

## ⚠️ Medical Disclaimer
**Important**: SAM AI is a diagnostic support tool designed for informational purposes and research. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the guidance of a qualified healthcare provider for any medical concerns.

## 📄 License
This project is licensed under the MIT License. See [LICENSE](https://opensource.org/licenses/MIT) for more details.
