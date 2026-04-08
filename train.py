import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- Load data ---
df = pd.read_csv("data.csv")
df.columns = df.columns.str.lower().str.strip()
print(f"Dataset loaded: {len(df)} rows")
print("Categories found:", df['category'].value_counts().to_dict())

# --- Preprocessing ---
df['symptoms'] = df['symptoms'].str.lower().str.strip()
X = df['symptoms']
y = df['category']

# --- Vectorize text to numbers ---
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# --- Split into train and test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42, stratify=y
)

# --- Train the model ---
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.1f}%")
print("\nDetailed report:")
print(classification_report(y_test, y_pred))

# --- Save model and vectorizer ---
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")

# --- Quick test ---
def predict(text):
    vec = vectorizer.transform([text.lower()])
    cat = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return cat, round(prob * 100, 1)

print("\n--- Test predictions ---")
tests = [
    "fever cough body ache",
    "chest pain left arm pain sweating",
    "sneezing runny nose watery eyes",
    "loss of taste dry cough fever",
    "severe headache nausea light sensitivity"
]
for t in tests:
    cat, conf = predict(t)
    print(f"  '{t}' => {cat} ({conf}% confidence)")