# Email Data Preprocessing for Machine Learning

## Credits and References

**Main Author:** Marco Antonio García Sánchez  
**Instructor and Course Base:** Santiago Hernández – Expert in Cybersecurity and Artificial Intelligence  
**Course:** "Machine Learning y Data Science: Curso Completo con Python" (Udemy)  
**Website:** techmind.ac  
**Course URL:** udemy.com/course/machine-learning-desde-cero/learn/lecture/19203700

### **Modifications and Own Contributions:**
- ✅ **Comprehensive preprocessing pipeline** with HTML cleaning and text normalization
- ✅ **Advanced exploratory data analysis** including discriminative words and pattern analysis
- ✅ **Professional documentation** with detailed explanations and insights
- ✅ **Enhanced code structure** for GitHub repository presentation
- ✅ **Integration planning** with machine learning projects

## Next Steps

This preprocessed dataset is ready for machine learning algorithms such as:
- **Logistic Regression** for binary classification
- **Naive Bayes** for text classification
- **Random Forest** for ensemble learning
- **Support Vector Machines** for high-dimensional text data

The processed data will be utilized in separate machine learning projects to demonstrate different classification approaches and compare algorithm performance.

## 📋 Project Description

This project demonstrates **comprehensive data preprocessing techniques** for email text data, transforming raw email content into machine learning-ready numerical features. The focus is on building a robust preprocessing pipeline for text classification tasks.

## Objectives

- **HTML Content Cleaning**: Remove HTML tags and extract meaningful text content
- **Text Preprocessing**: Implement punctuation removal, stopwords filtering, and stemming
- **Text Vectorization**: Convert preprocessed text into numerical features using CountVectorizer
- **Advanced Data Analysis**: Perform discriminative words analysis, spam pattern categorization, and language complexity analysis
- **Data Structure Preparation**: Organize data for machine learning algorithms

## Dataset

**TREC 2007 Public Spam Corpus**
- **75,419 emails** labeled as spam or ham
- **Real-world email data** with HTML and plain text formats
- **Balanced representation** of legitimate and spam content

## Implemented Methodology

### 1. **HTML Content Cleaning**
- Custom `MLStripper` class for HTML tag removal
- Support for both plain text and HTML email formats
- Preservation of meaningful text content

### 2. **Text Preprocessing Pipeline**
- **Punctuation removal** for cleaner text
- **Stopwords filtering** to remove common, uninformative words
- **Porter Stemming** to reduce words to root forms
- **Tokenization** for word-level analysis

### 3. **Text Vectorization**
- **CountVectorizer** for Bag-of-Words representation
- **Document-term matrix** with 20,890 unique features
- **Sparse matrix** format for memory efficiency

### 4. **Advanced Exploratory Data Analysis**
- **Discriminative Words Analysis**: Identify vocabulary that distinguishes spam from legitimate emails
- **Spam Pattern Analysis**: Categorize spam into financial, medical, and promotional patterns
- **Language Complexity Analysis**: Analyze sophistication differences between spam and ham emails

### 5. **Data Structure Preparation**
- Organized email format (subject + body)
- Maintained label associations for supervised learning
- Prepared numerical feature matrix ready for ML algorithms

## Key Results

- **Preprocessed dataset**: 1,000 emails with 20,890 unique features
- **Matrix sparsity**: ~99.5% (typical for text data)
- **Class distribution**: 78.3% spam, 21.7% ham
- **Feature analysis**: Identified discriminative words and spam patterns
- **Language insights**: Spam uses simpler language compared to legitimate emails

## Technologies Used

- **Python 3.x**
- **NLTK** - Natural language processing and stemming
- **Scikit-learn** - CountVectorizer and text preprocessing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **NumPy** - Numerical computations

## Key Concepts Demonstrated

### **Text Preprocessing**
- HTML parsing and content extraction
- Stopwords removal and stemming techniques
- Tokenization and text cleaning

### **Feature Engineering**
- Bag-of-Words representation
- Sparse matrix handling
- Text vectorization for ML algorithms

### **Exploratory Data Analysis**
- Discriminative word analysis
- Pattern recognition in text data
- Language complexity assessment

---

**Note:** This project is part of a Data Science and Machine Learning portfolio, demonstrating essential preprocessing skills for text classification tasks.
