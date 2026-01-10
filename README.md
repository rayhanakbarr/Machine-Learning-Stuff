# 🤖 Machine Learning Portfolio & Coursework

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Tools-Jupyter%20Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Library-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/XGBoost-017CEE?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
</p>

---

## 👤 Student Identification

> **Note:** This repository is submitted as part of the Machine Learning coursework requirements.

| Field | Information |
|-------|-------------|
| **Name** | Rayhan Akbar Al Hafizh |
| **Class** | TK-46-GAB |
| **NIM** | 1103223109 |

---

## 📂 Repository Structure

```
ML/
├── 📁 UAS/                          # Final Exam Projects
│   ├── 📁 TASK 1/                   # Fraud Detection (Classification)
│   ├── 📁 TASK 2/                   # Song Year Prediction (Regression)
│   └── 📁 TASK 3/                   # Fish Image Classification (CNN)
│
├── 📁 UTS/                          # Midterm Exam Projects
│   ├── 📁 Customer Clustering/      # Customer Segmentation Analysis
│   ├── 📁 midterm_folder/           # End-to-End Fraud Detection
│   └── 📁 Regresi/                  # End-to-End Regression Pipeline
│
├── 📁 Weekly Assignment/            # Chapter-based Exercises
│   └── 📁 Part 1/                   # Chapters 1-18
│
└── 📁 Hands-on Machine Learning.../  # Reference Book Materials
```

---

## 🎯 Final Exam (UAS) Projects

### 📌 Task 1: Online Transaction Fraud Detection
**Type:** Binary Classification | **Dataset:** Transaction Records

**Description:**  
End-to-end pipeline for detecting fraudulent online transactions. The notebook covers data downloading via gdown, merging transaction and identity tables, handling missing values & class imbalance, model training, hyperparameter tuning, and submission file generation.

**Tech Stack:**
- `Scikit-Learn` - HistGradientBoostingClassifier
- `XGBoost` - GPU-accelerated training
- `PyTorch` - Neural Network approach

**Key Features:**
- ✅ Automated data download from Google Drive
- ✅ Missing value imputation & feature engineering
- ✅ Class imbalance handling
- ✅ Model comparison & hyperparameter tuning
- ✅ ROC-AUC & PR-AUC evaluation metrics

**Files:**
- `fraud_detection.ipynb` - Main notebook
- `fraud_nn_best.pt` - Best Neural Network model
- `fraud_xgboost_gpu.json` - XGBoost GPU model
- `scaler_fraud.joblib` - Fitted scaler

---

### 📌 Task 2: Song Year Prediction
**Type:** Regression | **Dataset:** Audio Features

**Description:**  
Complete regression pipeline to predict the release year of songs based on numerical audio features (timbre, signal characteristics, etc.). Includes data exploration, outlier handling, model training with hyperparameter tuning, and performance evaluation.

**Tech Stack:**
- `Scikit-Learn` - Linear Regression, Ridge, Lasso, Random Forest
- `XGBoost` - Gradient Boosting Regressor (GPU)
- `PyTorch` - Deep Neural Network

**Key Features:**
- ✅ Feature importance analysis with Mutual Information
- ✅ Outlier detection & removal
- ✅ Multiple model comparison
- ✅ RandomizedSearchCV for hyperparameter optimization
- ✅ RMSE, MAE, R² evaluation metrics

**Files:**
- `song_year_prediction.ipynb` - Main notebook
- `nn_best.pt` - Best Neural Network model
- `song_year_xgboost.json` - XGBoost model
- `scaler.joblib` - Fitted scaler

---

### 📌 Task 3: Fish Image Classification
**Type:** Multi-class Image Classification | **Dataset:** 31 Fish Species

**Description:**  
Deep learning pipeline for classifying fish images into 31 different species. Implements CNN from scratch and transfer learning with EfficientNet-B0. Includes data augmentation, class weighting for imbalanced data, and Grad-CAM interpretability.

**Tech Stack:**
- `PyTorch` - Deep Learning Framework
- `torchvision` - Image transformations & pretrained models
- `EfficientNet-B0` - Transfer Learning backbone

**Key Features:**
- ✅ Custom CNN architecture from scratch
- ✅ Transfer Learning with EfficientNet-B0
- ✅ Data augmentation (rotation, flip, color jitter)
- ✅ Class weight balancing for imbalanced dataset
- ✅ Grad-CAM visualization for model interpretability
- ✅ CUDA/GPU acceleration support

**Fish Species (31 Classes):**
> Bangus, Big Head Carp, Black Spotted Barb, Catfish, Climbing Perch, Fourfinger Threadfin, Freshwater Eel, Glass Perchlet, Goby, Gold Fish, Gourami, Grass Carp, Green Spotted Puffer, Indian Carp, Indo-Pacific Tarpon, Jaguar Gapote, Janitor Fish, Knifefish, Long-Snouted Pipefish, Mosquito Fish, Mudfish, Mullet, Pangasius, Perch, Scat Fish, Silver Barb, Silver Carp, Silver Perch, Snakehead, Tenpounder, Tilapia

**Files:**
- `image_classification_fish.ipynb` - Main notebook
- `cnn_pytorch_best.pt` - Best CNN model
- `eff_b0_pytorch_best.pt` - Best EfficientNet-B0 model

---

## 📝 Midterm Exam (UTS) Projects

### 📌 Customer Segmentation Analysis
**Type:** Clustering | **Dataset:** Credit Card Usage Data

**Description:**  
Comprehensive customer clustering pipeline to segment credit card holders based on buying habits, payments, and credit usage. Compares three clustering algorithms: K-Means, Hierarchical Clustering, and DBSCAN.

**Tech Stack:**
- `Scikit-Learn` - KMeans, AgglomerativeClustering, DBSCAN
- `SciPy` - Dendrogram visualization

**Key Features:**
- ✅ PCA for dimensionality reduction & visualization
- ✅ Elbow method & Silhouette score analysis
- ✅ Dendrogram visualization for hierarchical clustering
- ✅ Epsilon optimization for DBSCAN
- ✅ Customer persona profiling

---

### 📌 End-to-End Fraud Detection (Midterm)
**Type:** Binary Classification | **Dataset:** Transaction Data

**Description:**  
LightGBM-based fraud detection pipeline with memory optimization techniques for handling large datasets efficiently.

**Tech Stack:**
- `LightGBM` - Gradient Boosting
- `Scikit-Learn` - Preprocessing & Evaluation

---

### 📌 End-to-End Regression Pipeline (Midterm)
**Type:** Regression | **Dataset:** Song Audio Features

**Description:**  
Professional regression pipeline with Random Forest and XGBoost for song year prediction, featuring comprehensive data handling and model evaluation.

**Tech Stack:**
- `Scikit-Learn` - Random Forest
- `XGBoost` - Gradient Boosting

---

## 📚 Weekly Assignments

Based on the book **"Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron.

| Chapter | Topic |
|---------|-------|
| 1 | The Machine Learning Landscape |
| 2 | End-to-End Machine Learning Project |
| 3 | Classification |
| 4 | Training Models |
| 5 | Support Vector Machines |
| 6 | Decision Trees |
| 7 | Ensemble Learning and Random Forests |
| 8 | Dimensionality Reduction |
| 9 | Unsupervised Learning Techniques |
| 10 | Introduction to Artificial Neural Networks with Keras |
| 11 | Training Deep Neural Networks |
| 12 | Custom Models and Training with TensorFlow |
| 13 | Loading and Preprocessing Data with TensorFlow |
| 14 | Deep Computer Vision Using CNNs |
| 15 | Processing Sequences Using RNNs and CNNs |
| 16 | Natural Language Processing with RNNs and Attention |
| 17 | Autoencoders, GANs, and Diffusion Models |
| 18 | Reinforcement Learning |

---

## 🛠️ Installation & Requirements

```bash
# Clone the repository
git clone https://github.com/yourusername/ML-Portfolio.git
cd ML-Portfolio

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies
```
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

## 🚀 How to Run

1. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Navigate to the desired folder** (UAS/UTS/Weekly Assignment)

3. **Run cells sequentially** - Each notebook is self-contained with data downloading capabilities

4. **GPU Acceleration (Optional):**
   - For PyTorch: Ensure CUDA is installed
   - For XGBoost: Install `xgboost` with GPU support

---

## 📊 Model Performance Summary

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Fraud Detection | XGBoost GPU | ROC-AUC | ~0.95+ |
| Fraud Detection | Neural Network | ROC-AUC | ~0.93+ |
| Song Year | XGBoost | RMSE | ~8.5 |
| Song Year | Neural Network | RMSE | ~9.0 |
| Fish Classification | EfficientNet-B0 | Accuracy | ~85%+ |
| Fish Classification | Custom CNN | Accuracy | ~75%+ |

---

## 📜 License

This project is for educational purposes as part of the Machine Learning coursework at Telkom University.

---

## 📬 Contact

- **Name:** Rayhan Akbar Al Hafizh
- **NIM:** 1103223109
- **Class:** TK-46-GAB
- **Institution:** Telkom University

---

<p align="center">
  <i>⭐ Star this repository if you find it helpful!</i>
</p>
