# Walmart Store Sales Prediction – Feature Selection & Dimensionality Reduction

**Author:** Marco Antonio García Sánchez  
**Repository:** [Data-Science-Projects](https://github.com/MarcoAntonioGarcia/Data-Science-Projects)  
**Path:** `6.-Machine Learning/4-Dimensionality Reduction/walmart_data_reduction_PCA-VIF-REF`

---

## Project Objective

Explore feature selection and dimensionality reduction techniques to prepare the dataset for predictive modeling of weekly Walmart sales, and evaluate their impact on a baseline regression model.

This notebook is developed as part of my Data Science and Machine Learning portfolio, demonstrating an end-to-end workflow:

Data exploration, cleaning, and preprocessing.
Feature engineering and transformation.
Regression modeling to predict weekly sales.
Evaluation and comparison of model performance using metrics such as R², RMSE, and MAE.
The project is based on the work of Yasser H, AI & ML Engineer at MediaAgility, Bengaluru, India. His original notebook can be found here: Walmart Sales Prediction – Kaggle.
---

## Dataset

- **Source:** [Kaggle – Walmart Store Sales](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)  
- **Content:** Historical weekly sales data for **45 stores** from 2010 to 2012  
- **Features:** Store ID, Department ID, Weekly Sales, CPI, Unemployment, Temperature, Fuel Price, Markdown 1–5, Date  
- **Target:** Weekly Sales (continuous variable)  

---

## Project Description

This notebook is part of my **Data Science and Machine Learning portfolio**, demonstrating an **end-to-end workflow**:

- Data exploration, cleaning, and preprocessing  
- Feature engineering and transformation  
- Regression modeling to predict weekly sales  
- Evaluation and comparison of model performance using **R², RMSE, and MAE**  

> Note: The project is based on the work of [Yasser H](https://www.kaggle.com/yasserh), AI & ML Engineer at MediaAgility, Bengaluru, India. This notebook is **fully understood, modified, and documented by me**, with additional insights and explanations for reproducibility.

---

## Methodology

### 1. Data Collection & Understanding

- Load dataset and inspect structure  
- Handle missing values and perform initial cleaning  

### 2. Exploratory Data Analysis (EDA)

- Analyze distribution of weekly sales  
- Examine correlation between features  
- Compare holiday vs non-holiday sales  

### 3. Preprocessing

- Encode categorical variables if needed  
- Scale numerical features  

### 4. Data Manipulation

- Prepare and clean data to ensure consistent format and remove noise  
- Facilitate feature engineering, scaling, and train/validation/test splits  
- Prevent sampling bias, data leakage, and overfitting  

### 5. Feature Selection / Extraction

- Identify relevant predictors  
- Remove irrelevant or redundant features  
- Prepare for dimensionality reduction  

### 6. Dimensionality Reduction

#### 6.1 Variance Inflation Factor (VIF)

- All VIF values below the usual threshold (~5) → low multicollinearity  
- No features removed  
- Limitation: Linear regression unable to capture the target relationship  
- Recommendation: Try complex or nonlinear models (Polynomial Regression, Decision Trees, Random Forest, Gradient Boosting)

#### 6.2 Recursive Feature Elimination (RFE)

- Applied RFE with linear regression to select the most relevant features  
- **Results:**  
  - R² Train: 0.146  
  - R² Test: 0.152  
  - RMSE Train: 524,837.68  
  - RMSE Test: 535,425.02  

#### 6.3 Principal Component Analysis (PCA)

- Reduced dataset dimensionality projecting features into orthogonal components  
- Selected **6 components** to explain ~90% of variance  
- **Results:**  
  - R² Train: 0.148  
  - R² Test: 0.154  
  - RMSE Train: 524,230.49  
  - RMSE Test: 534,788.59  

---

## Key Takeaways

- PCA, RFE, and VIF effectively reduce dimensionality and identify important features  
- Linear regression alone shows low predictive power (R² ≈ 0.15) → underfitting  
- Dimensionality reduction helps simplify the model but does not fully capture complex relationships  
- Future plans: Implement nonlinear models (Polynomial Regression, Decision Trees, Random Forest, Gradient Boosting) to improve predictive performance

---
