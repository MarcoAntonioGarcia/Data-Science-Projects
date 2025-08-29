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

#Content:

## Preprocessing

Emails need preprocessing to remove unnecessary noise:  
- HTML tags removal using a custom `MLStripper` class  
- Punctuation cleaning  
- Stopwords removal  
- Stemming to keep only the root of words  

This ensures the model only sees meaningful words from the email content.

---

## Vectorization

To convert emails to numerical features:  
1. **CountVectorizer**: transforms text into a feature matrix representing word counts.  
   - Each row = one email  
   - Each column = one word found across all emails  
   - Values = count of the word in the email  
2. **Optional OneHotEncoder**: explained but not strictly necessary; CountVectorizer is sufficient for this task.

---

## Training the Model

- A subset of 1,000 emails is used for initial training.  
- For full training, 10,000 emails are used with 2,000 emails reserved for testing.  
- The model is trained using Logistic Regression from `sklearn`.

---

## Testing and Evaluation

- Accuracy is evaluated on unseen emails (test set).  
- Example results:  
  - 1,000 emails → Accuracy: 1.000  
  - 12,000 emails → Accuracy: 0.988  

This demonstrates high predictive power and generalization.

---

## Conclusion

This project successfully implements a SPAM detection system:  
- Emails are cleaned and preprocessed for meaningful content  
- CountVectorizer effectively transforms text into numerical features  
- Logistic Regression achieves high accuracy even on large datasets  
- The workflow is modular, reproducible, and scalable  
- Ready for extension with additional datasets or more advanced models  

This notebook is included in the repository as a learning and portfolio exercise, demonstrating a full pipeline for a supervised machine learning problem.

---

## Credits

- Original course: “Machine Learning y Data Science: Curso Completo con Python” (Udemy)  
  URL: [udemy.com/course/machine-learning-desde-cero/learn/lecture/19203700](https://www.udemy.com/course/machine-learning-desde-cero/learn/lecture/19203700)  
- Author: Marco Antonio García Sánchez – modifications and improvements for GitHub portfolio
