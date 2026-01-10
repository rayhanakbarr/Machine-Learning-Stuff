# Machine Learning Portfolio & Coursework

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Tools-Jupyter_Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-yellow?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 👤 Student Identification
> **Note:** This repository is submitted as part of the Machine Learning coursework requirements.

* **Name:** [Rayhan Akbar Al Hafizh]
* **Class:** [TK-46-GAB]
* **NIM:** [1103223109]

---

## 📌 Repository Purpose
This repository serves as a comprehensive documentation of my studies in **Machine Learning**. It captures the progression from basic data manipulation to building complex, end-to-end machine learning pipelines.

The repository is structured to demonstrate proficiency in:
1.  **Supervised Learning:** Solving Regression and Classification problems.
2.  **Unsupervised Learning:** Implementing Clustering algorithms for data segmentation.
3.  **Model Evaluation:** Using appropriate metrics (Confusion Matrix, RMSE, Silhouette Score) to validate model performance.

---

## 📚 Weekly Assignments

The `Weekly Assignments` folder documents the step-by-step learning process:

* **Chapter 1-3:** Fundamentals of Python, NumPy arrays, Pandas DataFrames, and Data Visualization (Matplotlib/Seaborn).
* **Chapter 4-6:** Data Preprocessing, Cleaning, and Introduction to Supervised Learning algorithms.
* **Chapter 7-8:** Advanced topics, Model Tuning (GridSearch/RandomSearch), and Cross-Validation.

---

## 📂 Project Structure & Analysis (UTS)

The `UTS` (Mid-Term Exam) folder contains three major end-to-end projects. Below is the technical breakdown of the models and matrices used in each file:

### 1. 🕵️‍♂️ End-To-End Fraud Detection
* **File:** `End-To-End Fraud Detection.ipynb`
* **Objective:** Build a robust classifier to detect fraudulent transactions (binary classification) handling imbalanced datasets.
* **Key Techniques:**
    * **Data Handling:** Handling class imbalance (likely using SMOTE or Undersampling).
    * **Preprocessing:** StandardScaler for feature scaling.
* **Models Explored:**
    * Logistic Regression
    * Random Forest Classifier
    * Decision Tree
* **Evaluation Metrics:**
    * **Confusion Matrix:** To visualize False Positives vs False Negatives.
    * **Recall (Sensitivity):** Crucial for fraud detection to capture as many fraud cases as possible.
    * **F1-Score:** Balancing precision and recall.

### 2. 📈 End-To-End Regression Pipeline
* **File:** `End-To-End Regression Pipeline.ipynb`
* **Objective:** Predict continuous numerical values (e.g., prices, sales, or scores) based on feature inputs.
* **Key Techniques:**
    * **Pipeline Construction:** Using `sklearn.pipeline` to streamline preprocessing (Imputation, One-Hot Encoding) and modeling.
    * **Feature Engineering:** Handling categorical and numerical features automatically.
* **Models Explored:**
    * Linear Regression
    * Ridge / Lasso Regression (Regularization)
    * Random Forest Regressor
* **Evaluation Metrics:**
    * **RMSE (Root Mean Squared Error):** To measure the average magnitude of the errors.
    * **R2 Score:** To determine how well the model explains the variance in the data.

### 3. 👥 Customer Clustering Analysis
* **File:** `Customer_Clustering_Analysis.ipynb`
* **Objective:** Segment customers into distinct groups based on purchasing behavior or demographics (Unsupervised Learning).
* **Key Techniques:**
    * **EDA:** Visualizing data distribution using Pairplots and Heatmaps.
    * **Dimensionality Reduction:** (Optional) PCA (Principal Component Analysis) for 2D visualization.
* **Models Explored:**
    * **K-Means Clustering:** Partitioning data into $K$ distinct clusters.
    * **Hierarchical Clustering:** Building a dendrogram to understand data hierarchy.
* **Evaluation Metrics:**
    * **Elbow Method:** Used to determine the optimal number of clusters ($k$).
    * **Silhouette Score:** Assessing how well-separated the clusters are.

---

## 🚀 How to Navigate & Run

To replicate the analysis or run the notebooks locally:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/rayhanakbarr/Machine-Learning-Stuff.git](https://github.com/rayhanakbarr/Machine-Learning-Stuff.git)
    cd Machine-Learning-Stuff
    ```

2.  **Install Dependencies**
    Ensure you have the required Python libraries installed:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```

3.  **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
    ```

4.  **Explore**
    * Go to the `UTS` folder to see the major projects.
    * Open `End-To-End Fraud Detection.ipynb` to see the classification workflow.

---

<p align="center">
  <i>Created with ❤️ by Rayhan for Machine Learning Stuff.</i>
</p>
