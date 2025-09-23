# Walmart Store Sales Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project Overview

This project implements a **comprehensive machine learning solution** for predicting weekly sales at Walmart stores using advanced regression techniques. The analysis demonstrates professional-grade data science practices including feature engineering, model validation, and production deployment considerations.

**Author:** Marco Antonio García Sánchez  
**Objective:** Predict weekly sales for Walmart stores using machine learning regression models  
**Dataset:** [Kaggle – Walmart Store Sales](https://www.kaggle.com/datasets/yasserh/walmart-dataset)  
**Period:** Historical weekly sales data for **45 stores** from 2010 to 2012  
**Based on:** [Yasser H's Original Work](https://www.kaggle.com/code/yasserh/walmart-sales-prediction-best-ml-algorithms) with significant enhancements  

## Key Results

| Model | R² Test | RMSE Test | CV R² | Consistency | Production Ready |
|-------|---------|-----------|-------|-------------|------------------|
| **Lasso Regression** | 0.940 | 146,464 | 0.937 ± 0.010 | High | ✅ **Recommended** |
| **Polynomial Regression** | 0.953 | 125,883 | - | - | ⚠️ Moderate |
| **RFE + Linear** | 0.936 | 146,559 | - | - | ✅ Good |
| **PCA + Linear** | 0.835 | 235,843 | - | - | ⚠️ Limited |

## Quick Start

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
```

### Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn statsmodels
```

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd walmart-sales-prediction
```

2. **Download the dataset:**
   - Place `Walmart.csv` in https://www.kaggle.com/datasets/yasserh/walmart-dataset
   - Ensure the file path matches the notebook configuration

3. **Run the analysis:**
```bash
jupyter notebook walmart-sales-prediction.ipynb
```

## Methodology

### 1. Data Collection & Understanding
- Dataset quality verification (6,435 records, 10 features)
- Missing values and data consistency analysis
- Comprehensive data type analysis and basic statistics

### 2. Feature Engineering
- Temporal feature extraction (weekday, month, year)
- One-hot encoding for categorical variables
- Feature expansion from 10 to 68 variables

### 3. Exploratory Data Analysis (EDA)
- Target variable distribution with statistical normality tests
- Feature distribution visualization (histograms, boxplots)
- Correlation analysis between continuous variables
- Pattern identification and relationship analysis

### 4. Data Preprocessing
- Duplicate record removal (none found)
- Outlier detection and removal using IsolationForest algorithm
- Stratified train/validation/test split (70/15/15)
- Numerical feature standardization using StandardScaler

### 5. Feature Selection & Dimensionality Reduction
- **VIF Analysis**: Multicollinearity identification (VIF > 10)
- **RFE**: Automatic selection of 55 optimal features
- **PCA**: Dimensionality reduction to 50 components (90% variance)

### 6. Model Development & Evaluation
- **Linear Models**: Linear Regression, Lasso Regression with regularization
- **Non-linear Models**: Polynomial Regression (degree=2)
- **Performance Metrics**: R², RMSE evaluated on train/validation/test sets
- **Model Comparison**: Systematic evaluation of different approaches

### 7. Advanced Model Validation
- **Cross Validation**: 5-Fold CV for Lasso Regression
- **Learning Curves**: Bias-variance analysis and overfitting detection
- **Model Stability Assessment**: Comprehensive validation of generalization

## Key Findings

### Technical Insights
- **Feature Engineering Impact**: One-hot encoding improved performance by 600%
- **Regularization Effectiveness**: L1 regularization prevents overfitting
- **Temporal Features**: Weekday, month, year significantly enhance predictions
- **Model Stability**: Lasso shows excellent generalization across folds

### Business Insights
- **Store Location**: Significant impact on sales performance
- **Temporal Patterns**: Seasonal and weekly variations crucial
- **Economic Indicators**: CPI, Unemployment, Fuel Price correlations
- **Holiday Effects**: Holiday_Flag shows measurable impact

## Model Performance

### Lasso Regression (Recommended)
- **R² Score**: 0.940 (94% variance explained)
- **RMSE**: 146,464 (acceptable error margin)
- **Cross Validation**: 0.937 ± 0.010 (high consistency)
- **Learning Curves**: Excellent generalization with minimal overfitting
- **Production Ready**: ✅ Stable, reliable, interpretable

### Why Lasso is Optimal
- **Automatic Feature Selection**: L1 regularization eliminates irrelevant features
- **High Performance**: 94% R² with excellent consistency
- **Stable Generalization**: Low variance across different data splits
- **Interpretability**: Clear feature importance and coefficients
- **Production Ready**: Minimal overfitting, reliable predictions

## 🛠️ Technical Implementation

### Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
```

### Key Algorithms
- **Lasso Regression**: L1 regularization with automatic feature selection
- **Polynomial Regression**: Degree-2 non-linear modeling
- **Recursive Feature Elimination**: Optimal feature selection
- **Principal Component Analysis**: Dimensionality reduction
- **Isolation Forest**: Outlier detection and removal

## 📊 Visualizations

The project includes comprehensive visualizations:
- **Distribution Analysis**: Target variable and feature distributions
- **Correlation Heatmaps**: Feature relationship analysis
- **Model Performance**: Predicted vs actual sales plots
- **Cross Validation**: Performance across different data splits
- **Learning Curves**: Bias-variance analysis
- **Feature Importance**: Top contributing variables

## Production Deployment

### Implementation Strategy
1. **Model Deployment**: Use Lasso with α = 0.01 for production
2. **Feature Pipeline**: Implement standardized preprocessing pipeline
3. **Monitoring**: Set up performance tracking with R² and RMSE alerts
4. **Retraining**: Schedule monthly model updates with new data
5. **A/B Testing**: Compare predictions against actual sales for validation

### Risk Mitigation
- **Backup Model**: Keep Polynomial Regression as secondary option
- **Data Quality**: Implement automated data validation checks
- **Performance Monitoring**: Track model drift and accuracy degradation
- **Business Validation**: Regular review with domain experts

## Future Enhancements

### Model Improvements
- **Ensemble Methods**: Combine Lasso with Random Forest or XGBoost
- **Hyperparameter Tuning**: GridSearchCV for optimal α parameter
- **Advanced Features**: Lag features, rolling averages, interaction terms
- **Deep Learning**: Neural networks for complex non-linear patterns

### Data Enhancements
- **External Data**: Weather, events, competitor pricing
- **Real-time Features**: Current economic indicators, social media sentiment
- **Store-specific Data**: Local demographics, competition analysis
- **Temporal Extensions**: Longer historical data, seasonal adjustments

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 8GB RAM (recommended for large datasets)
- Jupyter Notebook or JupyterLab

### Python Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.12.0
jupyter>=1.0.0
```

## Author & Acknowledgments

**Marco Antonio García Sánchez**
- Data Science and Machine Learning Portfolio
- Professional implementation with comprehensive documentation

### Project Enhancements Over Original Work

This project is **based on the excellent work of [Yasser H](https://www.kaggle.com/yasserh)**, AI & ML Engineer at MediaAgility, but includes significant improvements and enhancements:

#### **Original Work (Yasser H):**
- Basic regression models (Linear, Lasso, Polynomial)
- Simple feature engineering
- Basic model evaluation

#### **Our Enhancements:**
- ✅ **Advanced Cross Validation**: 5-Fold CV with detailed analysis
- ✅ **Learning Curves**: Bias-variance analysis and overfitting detection
- ✅ **Comprehensive Feature Selection**: VIF, RFE, and PCA with detailed analysis
- ✅ **Statistical Rigor**: Normality tests, correlation analysis, residual analysis
- ✅ **Production Readiness**: Deployment strategy, monitoring, and risk mitigation
- ✅ **Professional Documentation**: Executive-level conclusions and business insights
- ✅ **Advanced Visualizations**: Learning curves, CV plots, and performance dashboards
- ✅ **Business Value Analysis**: ROI considerations and operational benefits
- ✅ **Future Enhancement Roadmap**: Technical infrastructure and model improvements

#### **Key Improvements:**
- **Model Validation**: From basic train/test to comprehensive CV and learning curves
- **Documentation**: From code comments to professional-grade documentation
- **Business Focus**: From technical implementation to business value and deployment
- **Statistical Analysis**: From basic metrics to rigorous statistical validation
- **Production Considerations**: From academic exercise to production-ready solution

---

This project demonstrates **senior-level data science competencies** including:
- ✅ Advanced feature engineering and selection
- ✅ Multiple modeling approaches with proper validation
- ✅ Statistical rigor and business acumen
- ✅ Production-ready implementation considerations
- ✅ Professional documentation and presentation
- ✅ **Significant enhancements over original baseline work**

*This analysis represents a complete end-to-end machine learning workflow suitable for professional deployment in retail analytics and demand forecasting applications.*
