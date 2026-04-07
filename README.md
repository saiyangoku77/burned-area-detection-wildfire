# 🔥 Burned Area Detection from Wildfire Satellite Images

A machine learning web application that detects burned areas in 
satellite images using Transfer Learning (MobileNetV2) and Flask.

**Validation Accuracy: 96.21%**

---

## 📁 Project Structure

burned_area_detection/
├── notebook.ipynb       
├── app.py               
├── requirements.txt      
├── templates/
│   └── index.html       
├── data/
│   ├── train/
│   │   ├── burned/       
│   │   └── not_burned/   
│   └── val/
│       ├── burned/       
│       └── not_burned/   
├── model/                
└── static/uploads/      

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/saiyangoku77/burned-area-detection-wildfire.git
cd burned-area-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🛰️ Dataset

Add your own satellite images to the data folders above.

Recommended sources: (This is what I used)
- [Kaggle Wildfire Datasets](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) 
Minimum recommended: **200+ images per class**

Split your images: **80% into train/** and **20% into val/**

---

## 🚀 How to Run

**Step 1 — Train the model**

Open `notebook.ipynb` in VS Code and run all cells top to bottom 
with `Shift + Enter`. Training saves the model automatically to 
`model/burned_area_model.h5`.

**Step 2 — Run the web app**
```bash
python app.py
```

**Step 3 — Open browser**
http://127.0.0.1:5000


Upload any satellite image, click **Analyze Image** and get your result.

---

## 🎯 Results

| Metric | Score |
|---|---|
| Validation Accuracy | 96.21% |
| Burned Precision | 98% |
| Burned Recall | 95% |
| Not Burned Recall | 97% |
| F1 Score | 0.96 |

---

## 🖥️ How It Works

1. User uploads a satellite image on the web app
2. Flask sends the image to the trained MobileNetV2 model
3. Model predicts: **Burned** or **Not Burned**
4. Result shown with confidence score
5. Image displayed with colored border:
   - 🔴 **Red border** = Burned area detected
   - 🟢 **Green border** = No burn damage found

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- Flask
- OpenCV
- PIL (Pillow)
- HTML / CSS / JavaScript

---

## ⚠️ Important Notes

- The `model/*.h5` file is not included — you must train it yourself
- Image folders are empty — add your own satellite images
- Do not rename the `burned` and `not_burned` folders
- Satellite images from the same sensor work best for consistency
