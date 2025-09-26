# K-Means Clustering for Bank Fraud Detection

## Credits and References

**Main Author:** Marco Antonio García Sánchez  
**Instructor and Course Base:** Santiago Hernández – Expert in Cybersecurity and Artificial Intelligence  
**Course:** "Machine Learning y Data Science: Curso Completo con Python" (Udemy)  
**Website:** techmind.ac  
**Course URL:** udemy.com/course/machine-learning-desde-cero/learn/lecture/19203700

## Project Description

This project implements **K-Means clustering** for exploratory analysis of credit card transaction data with the objective of identifying patterns and potential fraud indicators in banking. It uses **unsupervised learning** techniques to discover hidden structures in data without the need for prior labels.

## Objectives

- **Explore patterns** in credit card transactions through clustering
- **Identify clusters** that may contain fraudulent behaviors
- **Evaluate clustering quality** using multiple metrics
- **Analyze centroids** and distances between clusters
- **Differentiate** between clustering (exploratory) and classification (predictive)

## Dataset

**Credit Card Fraud Detection Dataset** - Kaggle
- **284,807 credit card transactions**
- **492 fraudulent cases** (0.172% of total)
- **Features V1-V28**: Principal components obtained through PCA
- **Time**: Time elapsed between transactions

## Implemented Methodology

### 1. **Exploratory Data Analysis (EDA)**
- Exploration of class distribution
- Analysis of main characteristics
- 2D and 3D visualizations

### 2. **Preprocessing**
- Considerations about already preprocessed data
- Class imbalance analysis
- Selection of relevant features

### 3. **K-Means Clustering**
- Implementation in 2D, 3D, and multidimensional spaces
- Optimization of number of clusters (Elbow Method)
- Convergence analysis

### 4. **Evaluation and Metrics**
- **Silhouette Score** (cohesion and separation)
- **Davies-Bouldin Index** (cluster quality)
- **Calinski-Harabasz Index** (variance ratio)
- **Centroids analysis** and distances

### 5. **Results Analysis**
- Identification of high-risk clusters
- Comparison with fraud distribution
- Advanced visualizations

## Main Results

- **Optimal number of clusters**: 2 (according to elbow method)
- **High-risk clusters identified** with fraud ratio > 5%
- **Centroids analysis** revealing behavior patterns
- **Quality metrics** confirming effective cluster separation

## Limitations and Considerations

- **Clustering ≠ Classification**: This analysis is exploratory, not predictive
- **Imbalanced data**: 0.172% fraud requires specialized techniques
- **PCA Features**: Limited interpretability due to transformation
- **Recommendation**: For real prediction, use supervised methods (Logistic Regression, Random Forest, XGBoost)

## Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical calculations
- **Scikit-learn** - K-Means and evaluation metrics
- **Matplotlib/Seaborn** - Visualizations
- **SciPy** - Distance analysis

## Clustering Concepts Explained

### **K-Means Clustering**
- Partitioning algorithm that minimizes intra-cluster sum of squares
- Requires specifying the number of clusters (k) a priori
- Sensitive to initialization and outliers

### **Evaluation Metrics**
- **Silhouette Score**: Measures how well an object fits its cluster vs others
- **Davies-Bouldin**: Ratio of intra-cluster dispersion vs inter-cluster separation
- **Calinski-Harabasz**: Ratio of variance between clusters vs variance within clusters

### **Applications in Fraud Detection**
- **Exploration**: Identify anomalous patterns in transactions
- **Segmentation**: Group similar behaviors
- **Risk analysis**: Identify clusters with higher fraud concentration

### **Modifications and Own Contributions:**
- ✅ **Complete methodology** redesigned for fraud detection
- ✅ **Advanced centroids analysis** with distance metrics
- ✅ **Additional metrics** (Davies-Bouldin, silhouette analysis per cluster)
- ✅ **High-risk cluster identification** with custom thresholds
- ✅ **Improved visualizations** with marked centroids
- ✅ **Professional conclusions** differentiating clustering from classification
- ✅ **Complete technical documentation** and coherent

## Recommended Next Steps

1. **Implement supervised methods** for real fraud prediction
2. **Outlier analysis** using specialized techniques
3. **Feature engineering** with temporal and behavioral variables
4. **Ensemble methods** combining multiple algorithms
5. **Temporal validation** considering the sequential nature of transactions


---

## Disclaimer

This project is for educational purposes only. The original methodology was adapted from academic sources with proper attribution. All modifications, enhancements, and original contributions are clearly documented above.

**Note:** This project is part of a Data Science and Machine Learning portfolio, demonstrating the application of clustering techniques to real banking fraud detection problems.
