# spam_detector.py
import os
import requests
import zipfile
import pandas as pd
import joblib
import streamlit as st
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

DATA_PATH = "sms_spam.csv"
NB_MODEL = "nb_model.pkl"
LR_MODEL = "lr_model.pkl"


# ============================================
# 1. DOWNLOAD + PREPARE DATA
# ============================================
def download_dataset():
    if os.path.exists(DATA_PATH):
        return

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    print("Downloading dataset...")
    r = requests.get(url)
    open("spam.zip", "wb").write(r.content)

    with zipfile.ZipFile("spam.zip", "r") as z:
        raw = z.read("SMSSpamCollection").decode("utf-8")

    rows = [line.split('\t') for line in raw.splitlines()]
    df = pd.DataFrame(rows, columns=["label", "text"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    df.to_csv(DATA_PATH, index=False)
    print("Dataset ready:", DATA_PATH)


# ============================================
# 2. TRAIN MODELS
# ============================================
def train_models():
    if os.path.exists(NB_MODEL) and os.path.exists(LR_MODEL):
        print("Models already trained.")
        return

    df = pd.read_csv(DATA_PATH)
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    nb = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("nb", MultinomialNB())
    ])

    lr = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("lr", LogisticRegression(max_iter=1000))
    ])

    nb.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    joblib.dump(nb, NB_MODEL)
    joblib.dump(lr, LR_MODEL)

    print("Models trained + saved.")


# ============================================
# 3. STREAMLIT UI
# ============================================
def run_ui():
    st.title("📩 Spam Detector — Naïve Bayes & Logistic Regression")

    nb = joblib.load(NB_MODEL)
    lr = joblib.load(LR_MODEL)

    mode = st.sidebar.radio("Choose mode", ["Predict message", "Batch sample"])

    if mode == "Predict message":
        text = st.text_area("Enter SMS message:")
        model_name = st.selectbox("Choose model:", ["Naïve Bayes", "Logistic Regression"])

        if st.button("Predict"):
            model = nb if model_name == "Naïve Bayes" else lr
            pred = model.predict([text])[0]
            prob = model.predict_proba([text])[0]

            st.write(f"### Prediction: **{'SPAM' if pred == 1 else 'HAM'}**")
            st.write(f"Probabilities — Ham: {prob[0]:.3f}, Spam: {prob[1]:.3f}")

    else:
        df = pd.read_csv(DATA_PATH).sample(10)
        st.write(df)

        if st.button("Predict sample"):
            df["NB"] = df["text"].apply(lambda x: nb.predict([x])[0])
            df["LR"] = df["text"].apply(lambda x: lr.predict([x])[0])
            st.write(df)


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    download_dataset()
    train_models()
    print("Run the UI using:  streamlit run spam_detector.py")
