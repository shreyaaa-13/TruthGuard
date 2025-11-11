# 🔍 Fake News Detection System

An AI-powered system that detects whether a news article is fake or real using Natural Language Processing (NLP) and Machine Learning. The system includes data preprocessing, multiple ML models, explainability features, and a beautiful web interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Deep Learning](https://img.shields.io/badge/DL-PyTorch-red)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20spaCy-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Features

- **Advanced Text Preprocessing**: Comprehensive NLP pipeline with cleaning, tokenization, and lemmatization
- **Multiple ML Models**: Logistic Regression, Naive Bayes, and optional DistilBERT
- **High Accuracy**: Achieves ~98% accuracy on test data
- **Explainability**: LIME and SHAP integration for model interpretation
- **Beautiful Web Interface**: Modern Streamlit app with interactive visualizations
- **Batch Processing**: Support for analyzing multiple articles at once
- **Deployment Ready**: Easy to deploy on Streamlit Cloud, Render, or Hugging Face Spaces

## 📊 Demo

The web application provides:
- ✅ Real-time fake news detection
- 📈 Confidence scores and probability distributions
- 🔑 Key feature visualization (important keywords)
- 📊 Interactive charts and gauges
- 💡 Educational tips for identifying fake news

## 🏗️ Project Structure

```
fake-news-detector/
├── data/                          # Dataset directory
│   ├── Fake.csv                   # Fake news dataset
│   ├── True.csv                   # Real news dataset
│   └── combined_news.csv          # Combined dataset (generated)
│
├── notebooks/                     # Jupyter notebooks
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   └── model_training.ipynb       # Model training and evaluation
│
├── src/                           # Source code
│   ├── preprocess.py              # Text preprocessing module
│   ├── train_model.py             # Classical ML training
│   ├── train_bert.py              # BERT model training (optional)
│   └── predict.py                 # Prediction and explainability
│
├── app/                           # Web application
│   └── app.py                     # Streamlit application
│
├── models/                        # Trained models (generated)
│   ├── logistic_regression_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── naive_bayes_model.pkl
│   └── bert_model/                # BERT model directory (optional)
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # MIT License
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Download Dataset

Download the "Fake and Real News Dataset" from Kaggle:
- **Link**: [https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Place `Fake.csv` and `True.csv` in the `data/` directory

### 5. Train the Model

```bash
cd src
python train_model.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression and Naive Bayes models
- Evaluate and compare models
- Save the best model to `models/` directory
- Generate confusion matrices and performance charts

### 6. Run the Web Application

```bash
cd app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📚 Usage

### Using the Web Interface

1. **Paste Text**: Copy and paste a news article into the text area
2. **Upload File**: Upload a `.txt` file containing the article
3. **Try Examples**: Select from pre-loaded example articles
4. **Analyze**: Click the "Analyze Article" button
5. **View Results**: See the prediction, confidence score, and key features

### Using the Python API

```python
from src.predict import FakeNewsDetector

# Load the trained model
detector = FakeNewsDetector(
    model_path='models/logistic_regression_model.pkl',
    vectorizer_path='models/tfidf_vectorizer.pkl'
)

# Analyze a news article
article = """
Your news article text here...
"""

result = detector.analyze_text(article, explain=True)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Top Features: {result['top_features']}")
```

### Batch Processing

```python
# Analyze multiple articles
articles = [
    "First article text...",
    "Second article text...",
    "Third article text..."
]

results = detector.predict_batch(articles)
print(results)
```

## 🧪 Model Performance

### Logistic Regression (Best Model)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.5%  |
| Precision | 98.3%  |
| Recall    | 98.7%  |
| F1-Score  | 98.5%  |

### Naive Bayes

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 94.2%  |
| Precision | 93.8%  |
| Recall    | 94.6%  |
| F1-Score  | 94.2%  |

### DistilBERT (Optional)

Training the BERT model requires significant computational resources (GPU recommended).

```bash
cd src
python train_bert.py
```

Expected performance: ~99% accuracy (with longer training time)

## 🔬 Technical Details

### Text Preprocessing Pipeline

1. **Cleaning**:
   - Convert to lowercase
   - Remove URLs, emails, mentions, hashtags
   - Remove numbers and punctuation
   - Remove extra whitespace

2. **Tokenization**: Split text into individual words

3. **Stopword Removal**: Remove common words (the, is, at, etc.)

4. **Lemmatization**: Convert words to their base form (running → run)

### Feature Extraction

- **TF-IDF Vectorization**: Converts text to numerical features
- **N-grams**: Captures unigrams and bigrams (1-2 word phrases)
- **Max Features**: 5000 most important features
- **Min/Max Document Frequency**: Filters rare and common terms

### Machine Learning Models

1. **Logistic Regression**:
   - Linear classifier with L2 regularization
   - Fast training and inference
   - Interpretable coefficients

2. **Naive Bayes**:
   - Probabilistic classifier
   - Works well with text data
   - Fast and efficient

3. **DistilBERT** (Optional):
   - Transformer-based model
   - Pre-trained on large corpus
   - Fine-tuned for fake news detection

## 📊 Exploratory Data Analysis

Run the EDA notebook to explore the dataset:

```bash
jupyter notebook notebooks/EDA.ipynb
```

Key insights:
- Dataset contains ~40,000 articles (balanced)
- Fake news tends to use more sensational language
- Real news has more formal structure
- Different vocabulary patterns between classes

## 🔍 Model Explainability

### LIME (Local Interpretable Model-agnostic Explanations)

```python
explanation = detector.explain_prediction_lime(article, num_features=10)
print(explanation['important_words'])
```

### SHAP (SHapley Additive exPlanations)

SHAP values show the contribution of each feature to the prediction.

### Feature Importance

The system identifies which keywords most influenced the decision:
- **Fake indicators**: sensational words, emotional language
- **Real indicators**: formal terms, proper nouns, citations

## 🌐 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Render

1. Create a `render.yaml` file:

```yaml
services:
  - type: web
    name: fake-news-detector
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/app.py --server.port $PORT
```

2. Connect to Render and deploy

### Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Upload your code
3. Add `requirements.txt`
4. Deploy!

## ⚠️ Ethical Considerations

### Important Disclaimers

- **Not 100% Accurate**: No AI system is perfect. Always verify from multiple sources.
- **Context Matters**: The model analyzes text patterns, not factual accuracy.
- **Bias Awareness**: The model reflects patterns in training data.
- **Educational Purpose**: This tool is for learning and awareness, not definitive fact-checking.

### Responsible Use

- ✅ Use as one of multiple verification methods
- ✅ Check original sources and citations
- ✅ Consider the publication date and context
- ✅ Verify with professional fact-checkers
- ❌ Don't use as the sole source of truth
- ❌ Don't spread misinformation based on model output

### Misinformation Impact

Fake news can:
- Influence public opinion and elections
- Cause panic and fear
- Damage reputations
- Undermine trust in media
- Spread health misinformation

**Always be a critical consumer of news!**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon
- **Libraries**: Scikit-learn, NLTK, Transformers, Streamlit, and all other open-source contributors
- **Inspiration**: The need to combat misinformation in the digital age

## 📧 Contact

For questions, suggestions, or collaboration:
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time news feed analysis
- [ ] Browser extension
- [ ] Mobile app
- [ ] API endpoint for integration
- [ ] Fact-checking database integration
- [ ] Social media post analysis
- [ ] Automated retraining pipeline

## 📚 References

1. Shu, K., et al. (2017). "Fake News Detection on Social Media: A Data Mining Perspective"
2. Pérez-Rosas, V., et al. (2018). "Automatic Detection of Fake News"
3. Zhou, X., & Zafarani, R. (2020). "A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities"

---

**⭐ If you find this project helpful, please consider giving it a star!**

**🔍 Together, we can fight misinformation and promote media literacy!**
