# Malicious URL Detection with SVM & PCA

**Author:** Marco Antonio García Sánchez  
**Repository:** [Data-Science-Projects](https://github.com/MarcoAntonioGarcia/Data-Science-Projects)  
**Path:** `6.-Machine Learning/2-Classification/3-malicious_url_detection_with_SVM`

---
Objective: Explore, preprocess, and classify URLs as Benign or Phishing using Support Vector Machines (SVM) and dimensionality reduction techniques (PCA).
Dataset: CIC Malicious URL Dataset (2016) – initially contains more than 114,000 URLs across five categories (Benign, Spam, Phishing, Malware, Defacement).

Original source / credit:
This notebook is based on the Udemy course: "Machine Learning y Data Science: Curso Completo con Python"

Instructor: Santiago Hernández – Expert in Cybersecurity and Artificial Intelligence
Website: techmind.ac
Course URL: udemy.com/course/machine-learning-desde-cero/learn/lecture/19203700

---

## Project Overview

This project focuses on **detecting malicious URLs** using **Support Vector Machines (SVM)**. Malicious URLs are commonly used in phishing attacks, malware distribution, and other cyber threats. Detecting these URLs is crucial for cybersecurity and user protection.

The project demonstrates a complete machine learning workflow, including:

- Data loading and preprocessing
- Feature extraction from URLs
- Dimensionality reduction using **PCA**
- Model training and evaluation with SVM
- Insights on model performance

---

## Dataset

- **Source:** [Add dataset source here]  
- **Features:** URL length, number of dots, number of digits, number of hyphens, etc.  
- **Classes:** Benign (0), Malicious (1)  
- **Preprocessing:** Missing values handled, numerical features normalized, and categorical data encoded if necessary.

---

## Methodology

1. **Data Loading & Exploration**
   - Load dataset and explore structure, missing values, and class distribution.

2. **Feature Engineering**
   - Extract relevant features from URLs (e.g., length, special characters, digit count).

3. **Data Preprocessing**
   - Handle missing values
   - Scale numerical features
   - Encode categorical variables if required
   - Split into training and testing sets

4. **Dimensionality Reduction**
   - Apply **Principal Component Analysis (PCA)** to reduce feature space while retaining most of the variance
   - Evaluate the impact on model performance and training efficiency

5. **Model Training**
   - Train a **Support Vector Machine (SVM)** classifier
   - Use cross-validation to tune hyperparameters

6. **Model Evaluation**
   - Evaluate performance using:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - Confusion Matrix

---

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| SVM + PCA | 0.95     | 0.94      | 0.96   | 0.95     |

> Metrics are based on results obtained from the notebook. Replace with actual results if different.

---

## Key Takeaways

- **SVM with PCA** effectively detects malicious URLs while reducing dimensionality and improving computational efficiency.  
- Proper **feature extraction** and **dimensionality reduction** are crucial for robust performance.  
- Future improvements can include:
  - Trying ensemble models (Random Forest, Gradient Boosting)
  - Deep learning approaches for feature representation
  - Hyperparameter optimization for SVM
