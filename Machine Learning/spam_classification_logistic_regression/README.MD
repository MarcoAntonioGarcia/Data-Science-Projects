# Spam Detection with Logistic Regression

**Author:** Marco Antonio García Sánchez  
**Objective:** Build a machine learning system to classify emails as SPAM or HAM using Logistic Regression.  
**Dataset:** [TREC 2007 Public Spam Corpus](http://plg.uwaterloo.ca/~gvcormac/trec07/) – 75,419 emails labeled as spam or ham.  

---

## Project Background

This project is based on a Udemy course exercise: **"Machine Learning y Data Science: Curso Completo con Python"**.  
URL: [udemy.com/course/machine-learning-desde-cero/learn/lecture/19203700](http://udemy.com/course/machine-learning-desde-cero/learn/lecture/19203700)  

The purpose of this notebook is to:

- Learn the end-to-end workflow for a supervised learning problem (SPAM detection)  
- Implement preprocessing, feature extraction, and model training  
- Serve as a **portfolio exercise** in my Machine Learning GitHub repository  

**Tools and Libraries Used:**

- Python 3.9  
- `scikit-learn` for machine learning algorithms and preprocessing  
- `nltk` for natural language processing (stopwords, stemming)  
- `pandas` for data inspection and visualization  

---

## Dataset

We use the **TREC 2007 Public Spam Corpus**, containing:

- 25,220 ham emails  
- 50,199 spam emails  

These emails were delivered to a particular server between April 8, 2007 and July 6, 2007.  

**Note:** The dataset is not included in this repository. You can download it manually from the [TREC 2007 corpus link](http://plg.uwaterloo.ca/~gvcormac/trec07/).

---

## Environment Setup

A dedicated Python environment is recommended for reproducibility:

```bash
conda create -n machine_learning_env python=3.9
conda activate machine_learning_env
pip install -r requirements.txt
