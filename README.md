# SAM AI — Medical Diagnosis Tool

A web app that helps predict the likelihood of certain diseases using machine learning. You fill in your health data, and SAM AI runs it through trained models to give you an assessment. It's not a replacement for a doctor — think of it more like a screening tool that can flag potential issues early.

🔗 **Try it out:** [sam-ai-7lwa.onrender.com](https://sam-ai-7lwa.onrender.com)
*(Heads up — it's on Render's free tier, so cold starts can take ~30s. Runs way faster locally.)*

---

## What It Can Detect

| Condition | How It Works |
|-----------|-------------|
| ❤️ Heart Disease | Evaluates risk factors like blood pressure, cholesterol, age, etc. |
| 🎗️ Breast Cancer | Analyzes cell measurements to classify tumors as benign or malignant |
| 🩸 Diabetes | Separate models for men and women, since risk factors differ |
| 🫁 Liver Disease | Checks enzyme levels and other liver function indicators |

## Other Features
- **User accounts** — sign up, log in, keep your results saved
- **Dashboard** — view and download your past test results
- **Health Q&A** — ask general health-related questions

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | React + TypeScript (Vite) |
| Backend | Flask (Python) |
| Database | SQLite (dev) / PostgreSQL (prod) |
| ML Models | scikit-learn |
| Hosting | Render |

---

## Run It Locally

**You'll need:** Node.js 18+ and Python 3.9+

```bash
# Clone it
git clone https://github.com/ARSHIYASHAFIZADE/SAM_Ai.git
cd SAM_Ai
```

**Backend:**
```bash
cd server
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py                # Starts on http://localhost:5000
```

**Frontend:**
```bash
cd sam
npm install
npm run dev                  # Starts on http://localhost:5173
```

---

## Screenshot

<img width="1883" height="896" alt="image" src="https://github.com/user-attachments/assets/6854a10d-2828-4c41-b5d3-c6407fa4cf9a" />
<img width="1212" height="903" alt="image" src="https://github.com/user-attachments/assets/21105ad4-a846-4ec4-bf76-9a079de225aa" />

---

## ⚠️ Disclaimer
This is a personal project built for learning and demonstration purposes. The predictions are based on ML models trained on public datasets — they are **not** medical advice. Always consult a real doctor for health concerns.

## License
MIT — do whatever you want with it. See [LICENSE](https://opensource.org/licenses/MIT).
