

# 🛡️ Skin Guard Pro – AI-Powered Skin Disease Detection System

A beginner-friendly AI web application that detects **7 common skin diseases** using an uploaded skin image and basic symptoms.  
Built with **Flask**, **TensorFlow**, and **OpenCV** using **transfer learning** on the HAM10000 dataset.

> ⚠️ **Disclaimer:** This project is for education and demonstration only.  
> It is **NOT** a medical diagnostic tool. Always consult a dermatologist for real medical decisions.

---

## 🚀 Features

- Upload a skin lesion image (JPG/PNG)
- Add symptoms such as *itching, redness*
- AI prediction using MobileNetV2 (Deep Learning)
- Symptom-based confidence boosting
- Severity estimation (Mild/Moderate/Severe)
- Smart recommendations and precautions
- Clean UI with instant results
- Fully offline — **no database required**

---

## 🧠 Diseases Detected (7 Classes)

| Code | Disease Name |
|------|--------------|
| akiec | Actinic Keratoses |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevi |
| vasc | Vascular Lesions |

---

## 🏗️ Tech Stack

| Layer | Technology |
|------|-----------|
| Backend | Python & Flask |
| AI Model | TensorFlow / Keras (MobileNetV2) |
| Image Processing | OpenCV, Pillow |
| Frontend | HTML, CSS |
| Dataset | HAM10000 (Kaggle) |

---

## 📂 Project Structure

```

skin-guard-pro/
│
├── app.py                   # Flask web server
├── train_model.py           # Train the model (optional)
├── skin_model.h5            # Trained model (required for prediction)
│
├── templates/
│   ├── index.html           # Upload page
│   └── results.html         # Result page
│
├── static/
│   ├── css/styles.css       # UI stylesheet
│   └── uploads/             # Temporary uploaded images
│
├── HAM10000/                # Dataset folder (only for training)
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Project
```bash
git clone https://github.com/yourusername/skin-guard-pro.git
cd skin-guard-pro
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install flask tensorflow opencv-python pillow numpy pandas scikit-learn
```

---

## 🧩 Model Setup

### Option A — Download Pretrained Model (Recommended)

Download from:
🔗 [https://github.com/ayoolaolafenwa/skin-cancer-detection/releases/download/v1.0/skin_cancer_model.h5](https://github.com/ayoolaolafenwa/skin-cancer-detection/releases/download/v1.0/skin_cancer_model.h5)

Rename to:

```
skin_model.h5
```

Place inside project root.

---

### Option B — Train Your Own Model

1️⃣ Download HAM10000 dataset
📌 [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

2️⃣ Put all images into:

```
HAM10000/all_images/
```

3️⃣ Train model:

```bash
python train_model.py
```

Training time: 30–60 minutes (CPU)

Output:

```
skin_model.h5
```

---

## ▶️ Run the Web Application

```bash
python app.py
```

Open in browser:
👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

Upload image → Enter symptoms → View AI result!

---

## 🧪 Testing

You can test using:

* Dataset images
* Google sample images like:

  * “melanoma skin lesion”
  * “basal cell carcinoma skin”

Better image clarity = better results ✔

---



Start command:

```
python app.py
```

---

## 🔮 Future Enhancements

* User login + history storage (SQLite/MongoDB)
* Advanced model (EfficientNet / ViT)
* Mobile-friendly UI / React frontend
* Doctor referral system
* Multi-language (English, Hindi, Marathi)
* Voice assistant with TTS

---





