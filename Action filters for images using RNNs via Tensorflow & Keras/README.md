# RNN Image Action Filters

A complete system for applying artistic image filters using **Recurrent Neural Networks (RNNs)** with TensorFlow/Keras, plus multiple traditional image-processing filters. Includes a full **Streamlit UI**, training utilities, and a standalone demo.

---

## Project Overview

This project shows how images can be processed **sequentially** (row-by-row) using **Bidirectional LSTMs**, instead of traditional CNNs.  
You can:

- Apply artistic filters (Sketch, Watercolor, Oil Painting, Cartoon, Vintage, Edge Enhance, Emboss)
- Use RNN models for image-to-image transformation
- Train RNNs to learn custom artistic styles
- Interact with everything through a Streamlit web app

---

## Quick Setup Guide

### **1️Create Project & Virtual Environment**
```bash
mkdir rnn-image-filters
cd rnn-image-filters

# Create venv
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

---

### **2️Install Dependencies**
```bash
pip install -r requirements.txt
```

---

### **3️Run the Application**
```bash
streamlit run image_filter_app.py
```

---

## Files Included

### **requirements.txt**
```
tensorflow==2.13.0
keras==2.13.1
streamlit==1.28.0
numpy==1.24.3
opencv-python==4.8.0.76
Pillow==10.0.0
matplotlib==3.7.2
scikit-image==0.21.0
```

---

### **rnn_filter_model.py**
Contains:

- `RNNImageFilter` → LSTM-based model for sequential row processing  
- `ImageFilterPresets` → 7 classical filters using OpenCV  
- `RNNFilterTrainer` → prepares data and trains RNN filters  

---

### **image_filter_app.py**
A fully interactive Streamlit UI with:

- Image upload  
- Filter selection  
- Side-by-side preview  
- RNN model training demo  
- Model info page  
- Download filtered image  

Run it using:
```bash
streamlit run image_filter_app.py
```

---

### **demo_filters.py**
A standalone script to apply **all filters** to one image and display/save the results.

Usage:
```bash
python demo_filters.py your_image.jpg
```

---

## Project Structure

```
rnn-image-filters/
├── venv/                     # Virtual environment
├── requirements.txt          # Dependencies
├── rnn_filter_model.py       # RNN model + artistic filters
├── image_filter_app.py       # Streamlit application
├── demo_filters.py           # Demo to apply all filters
└── README.md                 # Documentation
```

---

## Features

-  **Sketch Filter**
- **Watercolor Effect**
- **Oil Painting**
- **Cartoon Style**
- **Vintage Sepia**
- **Edge Enhancement**
- **Emboss 3D Effect**

Plus:

- **Bidirectional LSTM RNN** for sequential filtering  
- **Custom training pipeline**  
- **Streamlit UI** with real-time preview  
-  Download filtered images  
-  Model info page with architecture overview  

---

## Usage Tips

- Recommended image sizes: **256×256 to 1024×1024**
- RNN inference takes ~2–3 seconds on CPU
- For training a custom RNN:
  - Use **10+ images**
  - Prefer GPU (Google Colab recommended)

---

## Troubleshooting

### Oil painting filter missing?
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### TensorFlow GPU not found?
```bash
pip install tensorflow-gpu==2.13.0
```

### Out-of-memory during training?
- Reduce batch size
- Lower resolution (e.g., 128×128)
- Use fewer training samples

---

## Performance Summary

| Feature               | Result               |
|----------------------|----------------------|
| Filter speed         | ~2–3 sec/image       |
| Model size           | ~5 MB                |
| RAM usage            | ~1 GB                |
| Training time (GPU)  | 30–60 minutes        |

---

##  References

- TensorFlow 2.x Documentation  
- Keras LSTM / Bidirectional Layers  
- OpenCV Image Processing  
- Neural Style Transfer research  

---

##  License

This project is open for **educational and personal use**.  
Add your preferred license file (e.g., MIT) if publishing publicly.

---


