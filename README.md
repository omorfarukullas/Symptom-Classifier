# 🩺 Symptom Classifier

A simple AI-based medical symptom classification system that predicts possible disease categories based on user-provided symptoms.

---

## 🚀 Overview

The **Symptom Classifier** is a machine learning project designed to assist in basic health assessment.
Users can input symptoms (even with minor spelling mistakes), and the system predicts the most likely disease category.

This project is part of a larger vision for a smart healthcare system.

---

## 🎯 Features

* 🔍 Symptom-based disease prediction
* ✍️ Handles minor spelling mistakes in input
* 🤖 Machine learning-based classification
* 📊 Simple and lightweight model
* 🌐 Ready for integration with a web interface (Streamlit)

---

## 🧠 How It Works

1. User inputs symptoms (e.g., *fever, cough, headache*)
2. Input is cleaned and normalized
3. Symptoms are matched with trained dataset features
4. Model predicts the most probable disease category

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit (for UI)

---

## 📁 Project Structure

```
Symptom-Classifier/
│
├── data/               # Dataset files
├── model/              # Trained model files
├── app.py              # Streamlit UI
├── train_model.ipynb   # Model training notebook
├── utils.py            # Helper functions
├── requirements.txt    # Dependencies
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/omorfarukullas/Symptom-Classifier.git
cd Symptom-Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser and use the interface to input symptoms.

---

## 📌 Example

**Input:**

```
fevr, couh, hedache
```

**Output:**

```
Predicted Disease: Flu
```

---

## ⚠️ Disclaimer

This project is for educational purposes only.
It is **not a substitute for professional medical advice, diagnosis, or treatment**.

---

## 🔮 Future Improvements

* Improve model accuracy with larger dataset
* Add severity detection
* Integrate with hospital/doctor API
* Deploy as a full web application

---

## 👨‍💻 Author

**Omor Faruk Ullas**
CSE Undergraduate | AI & Web Development Enthusiast

---

## ⭐ Contributing

Contributions are welcome.
Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.
