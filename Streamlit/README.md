# Assignment 2 — Classification Models + Streamlit Deployment

## a) Problem Statement
Build and compare multiple ML classification models on a single dataset, evaluate them using standard metrics, and deploy an interactive Streamlit web app that allows dataset upload, model selection, and display of evaluation results.

## b) Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic) — UCI (binary classification)  
**Goal:** Predict whether a tumor is **malignant (0)** or **benign (1)** based on computed features from digitized images of fine needle aspirate (FNA) of breast mass.  
**Size:** 569 instances, 30 numeric features (>= 12 features, >= 500 instances satisfied)  
**Source:** UCI ML Repository (also available via `sklearn.datasets.load_breast_cancer()`)

## c) Models Used and Evaluation Metrics
Implemented 6 models on the same train/test split:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Comparison Table (Fill this after running `train_and_save_models.py`)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| KNN |  |  |  |  |  |  |
| Naive Bayes (Gaussian) |  |  |  |  |  |  |
| Random Forest (Ensemble) |  |  |  |  |  |  |
| XGBoost (Ensemble) |  |  |  |  |  |  |

### Observations on Model Performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline on linearly separable patterns; scaling improves stability and performance. |
| Decision Tree | Interpretable but can overfit; performance depends heavily on depth/regularization. |
| KNN | Sensitive to feature scaling; works well when local neighborhoods are informative. |
| Naive Bayes (Gaussian) | Fast baseline; assumes conditional independence, may underperform if features are correlated. |
| Random Forest (Ensemble) | Robust and typically strong; reduces overfitting vs single tree through bagging. |
| XGBoost (Ensemble) | Often best due to boosted trees capturing complex non-linearities; needs tuning but performs strongly. |

## How to Run Locally
```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Train and save all models + metrics
python model/train_and_save_models.py

# 3) Run Streamlit app
streamlit run app.py
