# 📱 UserInsight — Mobile Behaviour Predictor

> Enter your daily screen time and app usage — get your behaviour class, personalised tips, and risk warnings. Powered by a Random Forest Classifier trained to 98% accuracy.

---

## 🧩 Problem Statement

Mobile usage data is collected by every device, but users have no way to understand what their numbers actually *mean* for their wellbeing. There was no simple, interactive tool that could classify a person's digital behaviour pattern and give them actionable, personalised advice based on it.

UserInsight solves this — two inputs, one click, instant insight.

---

## 🎯 What It Does

1. User enters their **Screen On Time (hours/day)** and **App Usage Time (min/day)**
2. A trained **Random Forest Classifier** predicts their behaviour class (1–5)
3. The app displays their **behaviour label**, **personalised tips**, and **risk warnings**
4. Model accuracy is shown live in the sidebar on every run

---

## 🏷️ Behaviour Classes

| Class | Label | What It Means |
|---|---|---|
| 1 | **Low User** | Healthy balance — minimal screen dependency |
| 2 | **Moderate User** | Balanced — but watch for late-night usage creep |
| 3 | **High User** | Screen time limits recommended; risk of digital fatigue |
| 4 | **Very High User** | Consider a digital detox; eye strain and anxiety risk |
| 5 | **Extreme User** | Possible addiction symptoms — burnout and poor concentration |

---

## 📊 Dataset

**File:** `user_behavior_dataset.csv`  
**Records:** 700 users  
**Features:** 11 columns

| Column | Description |
|---|---|
| `User ID` | Unique identifier |
| `Device Model` | Phone model (e.g. Google Pixel 5, OnePlus 9) |
| `Operating System` | Android / iOS |
| `App Usage Time (min/day)` | Daily app usage in minutes (range: 30–598) |
| `Screen On Time (hours/day)` | Daily screen-on time in hours (range: 1–12) |
| `Battery Drain (mAh/day)` | Daily battery consumption |
| `Number of Apps Installed` | Total apps on device |
| `Data Usage (MB/day)` | Daily mobile data consumed |
| `Age` | User age |
| `Gender` | Male / Female |
| `User Behavior Class` | Target label (1–5) |

**Class distribution (balanced dataset):**

| Class | Count |
|---|---|
| 1 — Low User | 136 |
| 2 — Moderate User | 146 |
| 3 — High User | 143 |
| 4 — Very High User | 139 |
| 5 — Extreme User | 136 |

> The dataset is well-balanced across all 5 classes — no class imbalance handling was needed.

---

## 🤖 Model

**Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)  
**Features used:** 2 (`Screen On Time (hours/day)`, `App Usage Time (min/day)`)  
**Train / Test split:** 80% / 20% (`random_state=42`)  
**Accuracy:** **98%** on the test set  

The model trains automatically every time the app launches and displays its accuracy live in the sidebar:
```
✅ Model trained successfully (Accuracy: 98.00%)
```

---

## 📁 Project Structure

```
UserInsight/
├── app.py                      # Main Streamlit app — model, UI, prediction logic
├── user_behavior_dataset.csv   # Training dataset (700 records, 11 features)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Web Framework | Streamlit |
| ML Model | Scikit-learn (Random Forest) |
| Data Processing | Pandas, NumPy |
| Model Evaluation | `accuracy_score` from Scikit-learn |
| Visualisation | Matplotlib, Seaborn (EDA phase) |

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/UserInsight.git
cd UserInsight
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Update the dataset path

In `app.py`, update the CSV path to match your local machine:
```python
# Change this:
df = pd.read_csv(r"C:\Users\Aditi Sharma\Downloads\user_behavior_dataset.csv")

# To this (if CSV is in the same folder as app.py):
df = pd.read_csv("user_behavior_dataset.csv")
```

### 4. Run the app
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 🖥️ How to Use

1. Open the app in your browser
2. Enter your **Screen On Time** (hours per day) using the number input
3. Enter your **App Usage Time** (minutes per day)
4. Click **Predict**
5. Your behaviour class, tips, and risk warnings appear instantly

**Example:**
- Screen On Time: `6.4 hours` + App Usage: `393 min` → **Class 4: Very High User**
- Screen On Time: `1.5 hours` + App Usage: `60 min` → **Class 1: Low User**

---

## 📦 requirements.txt

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 🔮 Potential Improvements

- Add more input features (battery drain, number of apps, data usage) to improve prediction nuance
- Train with more ML algorithms and compare (XGBoost, SVM, KNN) using cross-validation
- Add a visualisation page showing where the user sits on the usage distribution
- Save predictions to a log file to track personal usage trends over time
- Deploy to Streamlit Cloud so users can access it without installing anything

---

## 👩‍💻 Author

**Aditi Sharma** — B.Tech CSE (Data Science), Class of 2026  
📧 aditias0101@gmail.com · [LinkedIn](https://linkedin.com/in/aditi-sharma-5a1a31289)
