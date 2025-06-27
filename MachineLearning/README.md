# 🤖 Machine Learning – ECMM422 Coursework

This section of the repository contains coursework from the ECMM422 Machine Learning module, divided into two main notebooks: `ca1.ipynb` and `ca2.ipynb`.

---

## 📘 `ca1.ipynb` – Supervised Learning & Data Analysis

This notebook focuses on classical supervised learning tasks applied to a structured dataset. It takes a complete ML workflow approach from data loading to evaluation.

### 🔧 Key Components

- **Exploratory Data Analysis (EDA)**:
    - Summary statistics
    - Data visualization using `matplotlib` and `seaborn`
    - Correlation heatmaps and feature selection

- **Data Preprocessing**:
    - Handling missing values
    - Feature scaling (standardization)
    - Train-test split

- **Supervised Models Implemented**:
    - Logistic Regression
    - k-Nearest Neighbors (k-NN)
    - Decision Trees
    - Random Forests

- **Model Tuning & Evaluation**:
    - Hyperparameter tuning (e.g. tree depth, `k`)
    - Confusion Matrix
    - Accuracy, Precision, Recall, F1-score

### 💬 Commentary

- Decision Trees showed overfitting tendencies without pruning.
- Random Forests offered improved stability and generalization.
- Logistic regression served as a strong baseline.
- Visual tools helped interpret class boundaries and performance.

---

## 📗 `ca2.ipynb` – Unsupervised Learning & SVM Classification

This notebook expands the scope to unsupervised learning and non-linear classification models, exploring deeper aspects of model evaluation and dimensionality reduction.

### 🔧 Key Components

- **Dimensionality Reduction**:
    - Principal Component Analysis (PCA)
    - Scree plot and explained variance analysis

- **Clustering**:
    - k-Means clustering
    - Evaluation with Silhouette Score and inertia

- **Support Vector Machines (SVM)**:
    - Linear and non-linear kernels
    - Decision boundary visualization
    - Classification report generation

- **Model Comparison**:
    - Accuracy and F1-score used to compare SVMs against previous classifiers
    - PCA + SVM pipeline tested for dimensionality-aware classification

### 💬 Commentary

- PCA effectively reduced redundancy and improved computation time.
- k-Means clusters showed moderate cohesion; cluster count tuning was key.
- SVM with RBF kernel outperformed linear classifiers on complex datasets.
- Visualizations highlighted model interpretability and clustering logic.

---

## 🚀 How to Run

```bash
cd UOE_CVML/MachineLearning
pip install pandas matplotlib seaborn scikit-learn
jupyter notebook
```

---

## 📬 Contact

Created by [@mihirm3hub](https://github.com/mihirm3hub)  
For coursework under the ECMM422 module – University of Exeter
