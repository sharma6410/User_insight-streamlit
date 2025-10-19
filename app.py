import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# 🎯 Title & Description
# -------------------------------
st.set_page_config(page_title="UserInsight", page_icon="📱", layout="centered")

st.title("📱 UserInsight — Predict User Behavior")
st.markdown("""
This app predicts **user behavior class** based on **Screen On Time** and **App Usage Time**.  
The model is trained using a Random Forest Classifier.
""")

# -------------------------------
# 📂 Load the dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"user_behavior_dataset.csv")
    return df

df = load_data()

# -------------------------------
# 🧠 Train Model
# -------------------------------
features = ['Screen On Time (hours/day)', 'App Usage Time (min/day)']
X = df[features]
y = df['User Behavior Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Accuracy
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.sidebar.success(f"Model trained successfully ✅ (Accuracy: {acc*100:.2f}%)")

# -------------------------------
# 📊 Define readable classes
# -------------------------------
class_info = {
    1: {
        "label": "Low User",
        "tips": "Good balance! Keep maintaining healthy phone habits.",
        "disadvantages": "May miss out on productivity or social updates if usage is too low."
    },
    2: {
        "label": "Moderate User",
        "tips": "You’re balanced — but be mindful of late-night usage.",
        "disadvantages": "Moderate users can sometimes slip into overuse without noticing."
    },
    3: {
        "label": "High User",
        "tips": "Try setting screen time limits and take regular breaks.",
        "disadvantages": "High users risk digital fatigue and lower sleep quality."
    },
    4: {
        "label": "Very High User",
        "tips": "Consider a digital detox or app usage tracker.",
        "disadvantages": "Too much usage can cause eye strain, anxiety, and productivity loss."
    },
    5: {
        "label": "Extreme User",
        "tips": "Strongly limit screen time — try focus or mindfulness apps.",
        "disadvantages": "Possible addiction symptoms, poor concentration, and burnout."
    }
}

# -------------------------------
# ⌨️ User Input
# -------------------------------
st.subheader("🔢 Enter your details:")

screen_time = st.number_input("Screen On Time (hours/day)", min_value=0.0, step=0.1, value=1.0)
app_usage = st.number_input("App Usage Time (min/day)", min_value=0.0, step=1.0, value=60.0)

# -------------------------------
# 🔮 Prediction
# -------------------------------
if st.button("Predict"):
    features_input = np.array([[screen_time, app_usage]])
    prediction = rf_model.predict(features_input)[0]

    if prediction in class_info:
        info = class_info[prediction]
        st.success(f"📊 Predicted Behavior: **{info['label']}**")
        st.write(f"💡 **Tips:** {info['tips']}")
        st.write(f"⚠️ **Disadvantages:** {info['disadvantages']}")
    else:
        st.warning(f"Predicted class: {prediction} (No info available)")

