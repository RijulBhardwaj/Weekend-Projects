# **Spam Detection System**

## **Project Overview**
This project is a complete Machine Learning application designed to classify SMS messages as either **“spam”** or **“ham”** (legitimate).  
It uses **Natural Language Processing (NLP)**, **Scikit-Learn**, and a dual-model system with **Naïve Bayes** and **Logistic Regression**.

A full **Streamlit web interface** allows users to train models, visualize metrics, and test custom messages.

---

## **Features**
- **Dual Machine Learning Models:**  
  Implements **Multinomial Naïve Bayes** and **Logistic Regression** for comparison.
- **TF-IDF Vectorization:**  
  Converts text messages into numerical features.
- **Interactive Streamlit UI:**  
  Train, evaluate, and test models directly in the browser.
- **Performance Metrics:**  
  Displays **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- **Visualizations:**  
  Confusion matrices, metric comparison graphs, and label distribution charts using **Seaborn** + **Matplotlib**.
- **Model Saving:**  
  Saves trained models and vectorizers to use later without retraining.

---

## **Technical Architecture**

```
.
├── spam_detector.py          # Main app: data, training, Streamlit UI  
├── spam_detector2.py         # Secondary testing script (optional)
├── main.py                   # Optional alternate execution script
├── nb_model.pkl              # Naïve Bayes trained model
├── lr_model.pkl              # Logistic Regression trained model
├── sms_spam.csv              # Prepared dataset
├── spam.zip                  # Raw downloaded dataset (auto-generated)
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── LICENSE                   # License file
```
---

## **Technologies Used**
- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**

---

## **Installation & Setup**

### **Prerequisites**
- Python **3.8+**

---

### **1. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

---

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## **Usage**

### **Run the Web Interface**
Start the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

This will open in your browser.

---

### **Application Workflow**
- **Home** – Overview of the system  
- **Train Models** – Loads data, preprocesses text, and trains both ML models  
- **Predict** – Enter text messages to classify as spam/ham with probabilities  
- **Model Comparison** – View accuracy, precision, recall, F1 score, and confusion matrices  

---

### **Run from Command Line (No UI)**

```bash
python main.py
```

---

## **Dataset Information**
The project currently uses a demo dataset inside `data_preparation.py`.

For production, use the **SMS Spam Collection Dataset** from UCI ML Repository.  
Update the function `load_and_prepare_data()` inside `data_preparation.py` to pull from a CSV or text file.

---

## **License**
This project is **open-source**, free for personal and educational use.

