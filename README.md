# Stroke Prediction

This project aims to predict the likelihood of a stroke occurrence in individuals based on various health and demographic features. The goal is to build a machine learning model that can assist in early detection and prevention strategies.

---

## 🎯 Objectives

- Analyze a dataset containing health-related features to identify patterns associated with stroke occurrences.  
- Preprocess and clean the data, handling missing values, encoding categorical variables, and scaling numerical features.  
- Train multiple classification models to predict stroke risk.  
- Evaluate model performance using appropriate metrics.  
- Interpret model results to derive actionable health insights.  

---

## 🧰 Tools & Technologies

- **Python**: Programming language used for data analysis and modeling.  
- **Jupyter Notebook**: Interactive environment for developing and documenting the analysis.  
- **Pandas**: Data manipulation and analysis.  
- **NumPy**: Support for large, multi-dimensional arrays and matrices.  
- **Scikit-learn**: Machine learning library for model building and evaluation.  
- **Matplotlib / Seaborn**: Data visualization libraries.  

---


---

## 📊 Dataset Overview

The dataset (`dataset.csv`) includes the following features:

- **Age**: Age of the individual.  
- **Hypertension**: Whether the individual has hypertension (1 = Yes, 0 = No).  
- **Heart Disease**: Whether the individual has heart disease (1 = Yes, 0 = No).  
- **Ever Married**: Marital status (Yes/No).  
- **Work Type**: Type of employment (e.g., Private, Self-employed, etc.).  
- **Residence Type**: Urban or Rural.  
- **Avg Glucose Level**: Average glucose level in the blood.  
- **Body Mass Index (BMI)**: Body Mass Index.  
- **Smoking Status**: Whether the individual smokes (formerly smoked, never smoked, smokes).  
- **Stroke**: Target variable indicating whether the individual has had a stroke (1 = Yes, 0 = No).  

---

## 🔄 Data Preprocessing

The following steps were performed to prepare the data for modeling:

- **Handling Missing Values**: Imputed or removed missing data as appropriate.  
- **Encoding Categorical Variables**: Applied one-hot encoding to categorical features like 'Work Type', 'Residence Type', and 'Smoking Status'.  
- **Feature Scaling**: Standardized numerical features such as 'Age', 'Avg Glucose Level', and 'BMI' to ensure uniformity in model training.  
- **Feature Selection**: Identified and retained relevant features based on correlation analysis and domain knowledge.  

---

## 🤖 Model Building & Evaluation

Multiple classification models were trained and evaluated:

- **Logistic Regression**  
- **Random Forest Classifier**  
- **Support Vector Machine (SVM)**  
- **K-Nearest Neighbors (KNN)**  

Model performance was assessed using:

- **Accuracy**: Proportion of correct predictions.  
- **Precision**: Proportion of positive predictions that are actually correct.  
- **Recall**: Proportion of actual positives that are correctly identified.  
- **F1-Score**: Harmonic mean of precision and recall.  
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.  

## ⚙️ New: End-to-End ML Pipeline (Production Version)

In addition to the Jupyter Notebook, a **Python-based pipeline** (`src/stroke_prediction.py`) has been implemented for automation and reproducibility.  
This script integrates **data cleaning**, **oversampling**, **feature preprocessing**, and **model training** into one executable workflow.
### 🔧 Pipeline Components

| Step | Description |
|------|--------------|
| **Data Cleaning** | Custom transformers fill missing smoking values, drop nulls, and remove “Other” gender entries. |
| **Oversampling** | Uses `RandomOverSampler` (from `imblearn`) with `sampling_strategy=0.5` to balance stroke/non-stroke data. |
| **Preprocessing** | Scales numeric features (`bmi`, `avg_glucose_level`) and one-hot encodes categorical ones. |
| **Model Training** | Trains a `RandomForestClassifier` with random_state=101. |
| **Cross-Validation** | Evaluates model with 5-fold CV on accuracy, precision, and recall. |
| **Model Saving** | Saves trained pipeline as `stroke_model.pkl` using `joblib`. |
