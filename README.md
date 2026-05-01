# AgroScan — Multi-Crop Disease and Health Detection

A deep learning system for automated plant disease detection across
5 crops and 30 disease classes, deployed as a real-time Streamlit web app.

---

## Results
| Metric | Score |
|---|---|
| Train Accuracy | 92.95% |
| Validation Accuracy | 85.39% |
| Test Accuracy | 85.67% |
| Classes | 30 |
| Images | 23,000+ |

---

## Crops Covered
Banana · Chilli · Radish · Groundnut · Cauliflower

---

## Tech Stack
TensorFlow · Keras · Streamlit · scikit-learn · Python · Google Colab

---

## Dataset
Multi-Crop Disease Dataset — Mendeley Data (CC BY 4.0)
DOI: 10.17632/6243z8r6t6.1

---

## How to Run
1. Download plant_disease_model.h5 from Google Drive (link below)
2. Place it in the same folder as app.py and class_names.json
3. Install dependencies:
pip install -r requirements.txt
4. Run the app:
streamlit run app.py

---

## Model Download
The trained model (64MB) is too large for GitHub.
Download from Google Drive: 
```
https://drive.google.com/file/d/1NbSYUb3uYw1WpqiON22MShp5u5BLSBKY/view?usp=sharing
```

---

## Architecture
```
3-block Vanilla CNN:
Input (128x128x3)
→ Conv2D(32) + ReLU + MaxPool
→ Conv2D(64) + ReLU + MaxPool
→ Conv2D(128) + ReLU + MaxPool
→ Dense(256) + Dropout(0.5)
→ Dense(30) + Softmax
```

---

## Project Structure
```
AgroScan/
├── README.md                        ← project overview
├── app.py                           ← Streamlit app
├── class_names.json                 ← class mappings
├── requirements.txt                 ← dependencies
├── .gitignore                       ← excludes large files
├── AgroScan_PlantDisease.ipynb      ← training notebook
└── results/
    ├── class_distribution.png
    ├── sample_images.png
    ├── loss_accuracy_curves.png
    └── confusion_matrix.png

```