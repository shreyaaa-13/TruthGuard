# 🚀 Quick Start Guide

Get up and running with the Fake News Detection System in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- Internet connection for downloading dependencies

## Installation Steps

### 1. Install Dependencies

```bash
# Run the automated setup script
python setup.py

# Or install manually
pip install -r requirements.txt
```

### 2. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 3. Get the Dataset

1. Visit [Kaggle Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
2. Download `Fake.csv` and `True.csv`
3. Place them in the `data/` directory

**Dataset Structure:**
```
data/
├── Fake.csv    # ~23,000 fake news articles
└── True.csv    # ~21,000 real news articles
```

### 4. Train the Model

```bash
cd src
python train_model.py
```

**Expected output:**
- Training time: 2-5 minutes
- Accuracy: ~98%
- Models saved to `models/` directory

### 5. Run the Web App

**Option A: Streamlit (Recommended)**
```bash
cd app
streamlit run app.py
```
Access at: http://localhost:8501

**Option B: Flask**
```bash
cd app
python flask_app.py
```
Access at: http://localhost:5000

## Usage Examples

### Web Interface

1. Open the app in your browser
2. Paste a news article or select an example
3. Click "Analyze Article"
4. View results with confidence scores and key features

### Python API

```python
from src.predict import FakeNewsDetector

# Load model
detector = FakeNewsDetector(
    'models/logistic_regression_model.pkl',
    'models/tfidf_vectorizer.pkl'
)

# Analyze text
article = "Your news article text here..."
result = detector.predict_single(article)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

### Batch Processing

```python
articles = [
    "First article...",
    "Second article...",
    "Third article..."
]

results = detector.predict_batch(articles)
print(results)
```

## Jupyter Notebooks

Explore the data and models interactively:

```bash
# Exploratory Data Analysis
jupyter notebook notebooks/EDA.ipynb

# Model Training and Evaluation
jupyter notebook notebooks/model_training.ipynb
```

## Troubleshooting

### Issue: Model not found

**Solution:**
```bash
cd src
python train_model.py
```

### Issue: NLTK data not found

**Solution:**
```python
import nltk
nltk.download('all')  # Downloads all NLTK data
```

### Issue: Memory error during training

**Solution:**
- Close other applications
- Reduce `max_features` in `train_model.py` (line 25)
- Use a smaller subset of data

### Issue: Streamlit not opening

**Solution:**
```bash
# Try specifying the port
streamlit run app.py --server.port 8502

# Or check if port 8501 is in use
netstat -ano | findstr :8501  # Windows
lsof -i :8501                 # Mac/Linux
```

## Next Steps

1. **Explore the notebooks** to understand the data and model
2. **Try different examples** in the web interface
3. **Customize the model** by adjusting parameters
4. **Deploy to cloud** using Streamlit Cloud or Render
5. **Integrate into your app** using the Python API

## Performance Tips

- **Faster predictions**: Use the classical ML model (Logistic Regression)
- **Better accuracy**: Train the BERT model (requires GPU)
- **Batch processing**: Use `predict_batch()` for multiple articles
- **Caching**: Models are loaded once and cached

## Common Commands

```bash
# Setup
python setup.py

# Train model
python src/train_model.py

# Run Streamlit app
streamlit run app/app.py

# Run Flask app
python app/flask_app.py

# Run notebooks
jupyter notebook

# Install new package
pip install package-name
pip freeze > requirements.txt
```

## File Structure

```
fake-news-detector/
├── data/              # Dataset files
├── models/            # Trained models
├── src/               # Source code
├── app/               # Web applications
├── notebooks/         # Jupyter notebooks
├── requirements.txt   # Dependencies
└── README.md          # Documentation
```

## Support

- **Documentation**: See [README.md](README.md)
- **Issues**: Check existing issues or create a new one
- **Questions**: Open a discussion on GitHub

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Train model | `python src/train_model.py` |
| Run Streamlit | `streamlit run app/app.py` |
| Run Flask | `python app/flask_app.py` |
| Run notebook | `jupyter notebook notebooks/EDA.ipynb` |
| Test prediction | `python src/predict.py` |

---

**Ready to detect fake news? Let's go! 🚀**
