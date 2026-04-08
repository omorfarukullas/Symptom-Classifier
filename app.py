import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from sklearn.ensemble import RandomForestClassifier

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Training.csv")

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

X = X.fillna(0)

# ---------------- TRAIN MODEL ----------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X, y)

# ---------------- MAPPING ----------------
normalize_map = {
    "fevr": "fever",
    "couh": "cough",
    "hedache": "headache",
    "head ache": "headache"
}

symptom_map = {
    "fever": "high_fever",
    "high fever": "high_fever",
    "mild fever": "mild_fever",
    "cough": "cough",
    "headache": "headache",
    "nausea": "nausea"
}

# ---------------- PREDICTION FUNCTION ----------------
def predict_disease(user_input):
    symptoms = [s.strip().lower() for s in user_input.split(",")]
    
    input_data = [0] * len(X.columns)
    matched = []

    for symptom in symptoms:
        original = symptom
        
        if symptom in normalize_map:
            symptom = normalize_map[symptom]
        
        if symptom in symptom_map:
            mapped = symptom_map[symptom]
            matched.append(f"{original} → {mapped}")
        
        else:
            match, score = process.extractOne(symptom, X.columns)
            
            if score > 60:
                mapped = match
                matched.append(f"{original} → {match}")
            else:
                matched.append(f"{original} → ❌")
                continue
        
        index = list(X.columns).index(mapped)
        input_data[index] = 1
    
    input_df = pd.DataFrame([input_data], columns=X.columns)
    
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    confidence = np.max(probs)

    return prediction, confidence, matched


# ---------------- UI ----------------
st.set_page_config(page_title="Disease Predictor", layout="centered")

st.title("🩺 Disease Prediction System")
st.write("Enter symptoms separated by commas (e.g., fever, cough, headache)")

user_input = st.text_input("Symptoms:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter symptoms!")
    else:
        prediction, confidence, matched = predict_disease(user_input)
        
        st.subheader("🧠 Prediction")
        st.success(prediction)
        
        st.subheader("📊 Confidence")
        st.info(f"{round(confidence * 100, 2)}%")
        
        st.subheader("🔍 Matched Symptoms")
        for m in matched:
            st.write(m)