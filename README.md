# TruthGuard — AI-Powered Fake News Detection System

TruthGuard is an AI-powered fake news detection system that combines Natural Language Processing (NLP), Machine Learning, and Explainable AI to classify news articles as real or fake.

The system includes advanced text preprocessing, multiple machine learning models, explainability techniques, and an interactive Streamlit web interface for real-time analysis.

---

## Problem Statement

The rapid spread of misinformation through digital media has made it increasingly difficult to distinguish between credible information and fake news.

This project aims to assist users in evaluating news credibility by leveraging machine learning and natural language processing techniques to identify linguistic patterns commonly associated with misinformation.

---

## Overview

TruthGuard analyzes news articles and predicts whether the content is likely to be real or fake.

The project includes:

* Text preprocessing and normalization
* Feature extraction using TF-IDF
* Multiple machine learning models
* Explainable AI integration
* Interactive web application
* Batch article analysis support

---

## Features

* Advanced NLP preprocessing pipeline
* Fake news classification using Machine Learning
* Logistic Regression and Naive Bayes models
* Optional DistilBERT implementation
* Explainability using LIME and SHAP
* Real-time article analysis
* Confidence score visualization
* Interactive Streamlit dashboard
* Batch prediction support

---

## Dataset

Dataset: Fake and Real News Dataset

Source:

https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

Dataset Contents:

* Fake News Articles
* Real News Articles
* News Headlines
* Article Content
* Source Metadata

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* spaCy
* PyTorch
* Streamlit
* LIME
* SHAP
* Jupyter Notebook

---

## Project Structure

```text
TruthGuard/
│
├── app/
│   └── app.py
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── train_bert.py
│   └── predict.py
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Methodology

### Data Preprocessing

The preprocessing pipeline performs:

* Lowercase conversion
* URL removal
* Punctuation removal
* Stopword removal
* Tokenization
* Lemmatization
* Text normalization

### Feature Extraction

Text is converted into numerical representations using:

* TF-IDF Vectorization
* Unigrams and Bigrams
* Feature Selection

### Model Training

The following models are implemented:

#### Logistic Regression

* Fast and efficient
* High interpretability
* Strong baseline performance

#### Naive Bayes

* Lightweight text classifier
* Effective for sparse text data

#### DistilBERT (Optional)

* Transformer-based architecture
* Context-aware language understanding
* Improved prediction performance

---

## Model Performance

### Logistic Regression

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 98.5% |
| Precision | 98.3% |
| Recall    | 98.7% |
| F1 Score  | 98.5% |

### Naive Bayes

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 94.2% |
| Precision | 93.8% |
| Recall    | 94.6% |
| F1 Score  | 94.2% |

---

## Explainable AI

TruthGuard includes explainability features to help users understand model predictions.

### LIME

Provides local explanations for individual predictions by highlighting influential words and phrases.

### SHAP

Measures the contribution of each feature toward the final prediction, improving transparency and trust.

---

## Skills Demonstrated

* Natural Language Processing (NLP)
* Text Classification
* Fake News Detection
* TF-IDF Vectorization
* Logistic Regression
* Naive Bayes
* DistilBERT
* Explainable AI
* Data Preprocessing
* Machine Learning
* Deep Learning
* Model Evaluation
* Streamlit Development
* Python Programming

---

## How to Run

### Clone the Repository

```bash
git clone https://github.com/shreyaaa-13/TruthGuard.git
cd TruthGuard
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK Resources

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Train the Model

```bash
python src/train_model.py
```

### Launch the Application

```bash
streamlit run app/app.py
```

---

## Future Enhancements

* Multi-language fake news detection
* Real-time news feed monitoring
* Browser extension support
* API integration
* Mobile application
* Automated retraining pipeline
* Advanced transformer-based models

---

## Ethical Considerations

TruthGuard is intended as an educational and analytical tool.

Important notes:

* Predictions are probabilistic and not definitive facts.
* Users should verify information through trusted sources.
* AI-generated predictions should not replace professional fact-checking.
* Biases present in training data may influence results.

---


Artificial Intelligence and Data Science Student
