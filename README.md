<div align="center">

# 🤖 Machine Learning Portfolio & Coursework

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=6C63FF&center=true&vCenter=true&width=600&lines=End-to-End+ML+Pipelines;Deep+Learning+with+PyTorch;GPU+Accelerated+Training;Classification+%7C+Regression+%7C+Clustering" alt="Typing SVG" />

<br>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-017CEE?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

</div>

---

## 👤 Student Identification

> **Note:** This repository is submitted as part of the Machine Learning coursework requirements.

| | |
|:---:|:---|
| 📛 **Name** | Rayhan Akbar Al Hafizh |
| 🎓 **Class** | TK-46-GAB |
| 🆔 **NIM** | 1103223109 |

---

## 📂 Repository Structure

```
📦 ML/
├── 📁 UAS/                                                 
│   ├── 📁 TASK 1: FRAUD DETECTION/                         
│   │   ├── 📓 Fraud Detection.ipynb
│   │   ├── 🧠 fraud_nn_best.pt
│   │   ├── 🌲 fraud_xgboost_gpu.json
│   │   └── 📊 data/
│   ├── 📁 TASK 2: REGRESSION - SONG YEAR PREDICTION/       
│   │   ├── 📓 song year prediction.ipynb
│   │   ├── 🧠 nn_best.pt
│   │   ├── 🌲 song_year_xgboost.json
│   │   └── 📊 data_regression/
│   └── 📁 TASK 3: IMAGE CLASSIFICATION/                    
│       ├── 📓 image_classification_fish.ipynb
│       ├── 🧠 cnn_pytorch_best.pt
│       ├── 🧠 eff_b0_pytorch_best.pt
│       └── 🖼️ train/ val/ test/
│
├── 📁 UTS/                                                 
│   ├── 📓 Customer_Clustering_Analysis.ipynb
│   ├── 📓 End-To-End Fraud Detection.ipynb
│   ├── 📓 End-To-End Regression Pipeline.ipynb
│   └── 📊 submission for fraud detection.csv
│
└── 📁 Weekly Assignments/                                    
    └── 📓 Chapter 1-18.ipynb
```

---

## 🎯 Final Exam (UAS) Projects

<table>
<tr>
<td width="33%">

### 🔍 Task 1: Fraud Detection
**Classification** | GPU Accelerated

<img src="https://img.shields.io/badge/Type-Binary%20Classification-blue?style=flat-square">
<img src="https://img.shields.io/badge/GPU-CUDA-green?style=flat-square">

</td>
<td width="33%">

### 🎵 Task 2: Song Year Prediction
**Regression** | GPU Accelerated

<img src="https://img.shields.io/badge/Type-Regression-purple?style=flat-square">
<img src="https://img.shields.io/badge/GPU-CUDA-green?style=flat-square">

</td>
<td width="33%">

### 🐟 Task 3: Fish Classification
**Image Classification** | Deep Learning

<img src="https://img.shields.io/badge/Type-Multi--Class-orange?style=flat-square">
<img src="https://img.shields.io/badge/CNN-PyTorch-red?style=flat-square">

</td>
</tr>
</table>

---

### 📌 Task 1: Online Transaction Fraud Detection

> **End-to-End Classification Pipeline dengan GPU Acceleration**

Sistem deteksi fraud untuk transaksi online menggunakan multiple models dengan akselerasi GPU CUDA.

| Model | Framework | Hardware |
|-------|-----------|----------|
| Logistic Regression | Scikit-Learn | CPU (Baseline) |
| XGBoost | XGBoost | **GPU CUDA** |
| Neural Network | PyTorch | **GPU CUDA** |

**🔧 Tech Stack:**
```
scikit-learn • xgboost • pytorch • pandas • seaborn • gdown
```

**✨ Key Features:**
- ✅ Automated data download via Google Drive
- ✅ Handling class imbalance dengan class weights
- ✅ Feature engineering & preprocessing
- ✅ Hyperparameter tuning dengan RandomizedSearchCV
- ✅ Evaluation: ROC-AUC, PR-AUC, Confusion Matrix

**📁 Output Files:**
| File | Description |
|------|-------------|
| `fraud_nn_best.pt` | Best Neural Network model |
| `fraud_xgboost_gpu.json` | XGBoost GPU model |
| `scaler_fraud.joblib` | Fitted StandardScaler |
| `submission_gpu.csv` | Final predictions |

---

### 📌 Task 2: Song Year Prediction

> **End-to-End Regression Pipeline dengan GPU Acceleration**

Memprediksi tahun rilis lagu berdasarkan fitur-fitur audio numerik (timbre, karakteristik sinyal musik).

| Model | Framework | Hardware |
|-------|-----------|----------|
| Linear Regression | Scikit-Learn | CPU (Baseline) |
| XGBoost | XGBoost | **GPU CUDA** |
| Neural Network | PyTorch | **GPU CUDA** |

**🔧 Tech Stack:**
```
scikit-learn • xgboost • pytorch • pandas • matplotlib • gdown
```

**✨ Key Features:**
- ✅ Exploratory Data Analysis (EDA)
- ✅ Outlier detection & handling
- ✅ Feature scaling dengan StandardScaler
- ✅ Model comparison (Linear vs XGBoost vs NN)
- ✅ Evaluation: RMSE, MAE, R² Score

**📁 Output Files:**
| File | Description |
|------|-------------|
| `nn_best.pt` | Best Neural Network model |
| `song_year_xgboost.json` | XGBoost model |
| `scaler.joblib` | Fitted StandardScaler |

---

### 📌 Task 3: Fish Image Classification

> **Deep Learning Pipeline dengan CNN & Transfer Learning**

Klasifikasi gambar ikan ke dalam **31 spesies** menggunakan Custom CNN dan EfficientNet-B0.

| Model | Architecture | Method |
|-------|-------------|--------|
| Custom CNN | Conv2D + MaxPool | From Scratch |
| EfficientNet-B0 | Pretrained | Transfer Learning |

**🐟 31 Fish Species:**
<details>
<summary>Click to expand species list</summary>

```
Bangus • Big Head Carp • Black Spotted Barb • Catfish • Climbing Perch
Fourfinger Threadfin • Freshwater Eel • Glass Perchlet • Goby • Gold Fish
Gourami • Grass Carp • Green Spotted Puffer • Indian Carp • Indo-Pacific Tarpon
Jaguar Gapote • Janitor Fish • Knifefish • Long-Snouted Pipefish • Mosquito Fish
Mudfish • Mullet • Pangasius • Perch • Scat Fish • Silver Barb • Silver Carp
Silver Perch • Snakehead • Tenpounder • Tilapia
```

</details>

**🔧 Tech Stack:**
```
pytorch • torchvision • efficientnet • matplotlib • seaborn
```

**✨ Key Features:**
- ✅ Data Augmentation (rotation, flip, color jitter)
- ✅ Class weight balancing untuk imbalanced dataset
- ✅ Transfer Learning dengan EfficientNet-B0
- ✅ **Grad-CAM** visualization untuk interpretability
- ✅ Early stopping & model checkpointing

**📁 Output Files:**
| File | Description |
|------|-------------|
| `cnn_pytorch_best.pt` | Best Custom CNN model |
| `eff_b0_pytorch_best.pt` | Best EfficientNet-B0 model |

---

## 📝 Midterm Exam (UTS) Projects

<table>
<tr>
<td width="33%">

### 🎯 Customer Clustering
**Unsupervised Learning**

Segmentasi pelanggan kartu kredit berdasarkan perilaku transaksi.

**Algorithms:**
- K-Means
- Hierarchical
- DBSCAN

</td>
<td width="33%">

### 🔍 Fraud Detection
**Classification**

Pipeline deteksi fraud dengan LightGBM dan optimasi memori.

**Highlights:**
- Memory optimization
- Feature engineering
- LightGBM

</td>
<td width="33%">

### 📈 Regression Pipeline
**Regression**

End-to-end regression untuk prediksi tahun lagu.

**Models:**
- Random Forest
- XGBoost

</td>
</tr>
</table>

---

## 📚 Weekly Assignments

Berdasarkan buku **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron.

| Chapter | Topic | Status |
|:-------:|-------|:------:|
| 1 | The Machine Learning Landscape | ✅ |
| 2 | End-to-End Machine Learning Project | ✅ |
| 3 | Classification | ✅ |
| 4 | Training Models | ✅ |
| 5 | Support Vector Machines | ✅ |
| 6 | Decision Trees | ✅ |
| 7 | Ensemble Learning and Random Forests | ✅ |
| 8 | Dimensionality Reduction | ✅ |
| 9 | Unsupervised Learning Techniques | ✅ |
| 10 | Introduction to ANNs with Keras | ✅ |
| 11 | Training Deep Neural Networks | ✅ |
| 12 | Custom Models with TensorFlow | ✅ |
| 13 | Loading & Preprocessing Data | ✅ |
| 14 | Deep Computer Vision (CNNs) | ✅ |
| 15 | Processing Sequences (RNNs) | ✅ |
| 16 | NLP with RNNs and Attention | ✅ |
| 17 | Autoencoders, GANs, Diffusion | ✅ |
| 18 | Reinforcement Learning | ✅ |

---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ML-Portfolio.git
cd ML-Portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 📦 Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
torch>=2.0.0
torchvision>=0.15.0
gdown>=4.5.0
joblib>=1.1.0
```

---

## 🚀 Quick Start

```python
# 1. Open Jupyter Notebook
jupyter notebook

# 2. Navigate to desired folder (UAS/UTS/Weekly Assignment)

# 3. Run cells sequentially - each notebook is self-contained

# 4. For GPU acceleration, ensure CUDA is properly installed
```

---

## 📊 Performance Summary

| Task | Best Model | Metric | Score |
|------|------------|--------|-------|
| 🔍 Fraud Detection | XGBoost GPU | ROC-AUC | ~0.95 |
| 🎵 Song Year | XGBoost GPU | RMSE | ~8.5 |
| 🐟 Fish Classification | EfficientNet-B0 | Accuracy | ~85% |

---

<div align="center">

## 📬 Contact

**Rayhan Akbar Al Hafizh**  
📧 NIM: 1103223109 | 🎓 Class: TK-46-GAB

---

<img src="https://img.shields.io/badge/Institution-Telkom%20University-red?style=for-the-badge">

<br>

⭐ **This repository is made with love** ⭐

</div>
