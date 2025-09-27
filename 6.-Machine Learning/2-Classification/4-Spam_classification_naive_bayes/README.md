# Spam Classification with Naive Bayes Algorithm

## Credits and Attribution

### Original Source and Inspiration
This project is based on the comprehensive course **"Machine Learning y Data Science: Curso Completo con Python"** by [Udemy](https://www.udemy.com/course/machine-learning-desde-cero/learn/lecture/19203700).

The foundational concepts, methodology, and initial implementation approach were learned from this excellent educational resource, which provided the essential framework for understanding:
- Text preprocessing techniques for email classification
- Naive Bayes algorithm implementation
- Data preprocessing pipelines for machine learning
- Performance evaluation methodologies

### Author's Contributions and Modifications
**Marco Antonio García Sánchez** has made significant enhancements and modifications to the original coursework

## Project Description

This project implements a **spam classifier** using the **Naive Bayes algorithm** to distinguish between legitimate emails (ham) and spam. The project demonstrates a complete machine learning pipeline, from text preprocessing to model evaluation with advanced metrics.

### Objectives

- Implement the Naive Bayes algorithm for binary email classification
- Demonstrate advanced text preprocessing techniques
- Evaluate model performance with multiple metrics
- Analyze discriminative features between spam and legitimate emails

## Dataset

**TREC 2007 Public Spam Corpus**
- **Total emails:** 75,419
- **Spam:** 50,199 emails (66.5%)
- **Ham (legitimate):** 25,220 emails (33.5%)
- **Period:** April - July 2007
- **Format:** Emails with complete headers and HTML/text content

## Key Features

### Advanced Preprocessing
- **HTML Cleaning:** Automatic removal of HTML tags
- **Noise Filtering:** Removal of punctuation and stopwords
- **Stemming:** Word reduction to root forms (Porter Stemmer)
- **Vectorization:** Transformation to Bag-of-Words representation

### Naive Bayes Algorithm
- **Probabilistic classification** based on word frequencies
- **Class imbalance handling** (78.3% spam vs 21.7% ham)
- **Interpretability** for understanding classification decisions

### Exploratory Analysis
- **Class distribution** with visualizations
- **Discriminative word analysis** (15 spam-exclusive words)
- **Email length patterns** by category
- **Detailed performance metrics**

## Model Results

| Metric | Value |
|---------|-------|
| **Accuracy** | 97.18% |
| **F1-Score** | 94.6% |
| **Precision** | 94.4% |
| **Recall** | 94.8% |

### Most Discriminative Words (Spam-Only)
1. **viagra** - 346,000x more frequent in spam
2. **anatrim** - 342,000x more frequent in spam
3. **ciali** - 315,000x more frequent in spam
4. **pharmaci** - 160,000x more frequent in spam
5. **dose** - 117,000x more frequent in spam


### 2. Main Notebook Sections

#### **Sections 1-2: Setup and Cleaning**
- Python environment configuration
- HTML cleaning class implementation
- Text preprocessing pipeline

#### **Section 3: Preprocessing Demonstration**
- Individual email cleaning example
- Vectorization with CountVectorizer
- Transformation to numerical format

#### **Sections 4-5: Data Preparation**
- Loading and processing 1,000 emails
- Feature matrix creation (20,890 features)
- Exploratory data analysis

#### **Section 6: Advanced Analysis**
- Class distribution
- Discriminative word analysis
- Spam pattern visualizations

#### **Section 7: Classification and Evaluation**
- Naive Bayes implementation
- Performance metrics
- Results and conclusions

## Technical Methodology

### Preprocessing Pipeline
1. **HTML Cleaning:** `MLStripper` class for pure text extraction
2. **Tokenization:** Separation into individual words
3. **Filtering:** Removal of punctuation and stopwords
4. **Stemming:** Reduction to root forms with Porter Stemmer
5. **Vectorization:** CountVectorizer for numerical representation

### Naive Bayes Algorithm
- **Independence assumption** between features
- **Probabilistic modeling** based on frequencies
- **Laplace smoothing** to avoid zero probabilities
- **Binary classification** spam vs ham

## Included Visualizations

- **Class distribution** (pie chart)
- **Email length by category** (histograms)
- **Top discriminative words** (horizontal bar charts)
- **Performance metrics** (pie charts)
- **Precision vs recall analysis**

## Practical Applications

### Use Cases
- **Spam filters** for mail servers
- **Content classification** systems
- **Phishing detection** in emails
- **Sentiment analysis** in text

### Approach Advantages
- **High accuracy** (97.18% accuracy)
- **Fast processing** for large volumes
- **Decision interpretability**
- **Scalability** for large datasets



