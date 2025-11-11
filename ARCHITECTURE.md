# 🏗️ System Architecture

## Overview

The Fake News Detection System follows a modular, layered architecture designed for scalability, maintainability, and ease of deployment.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐              ┌──────────────────┐        │
│  │  Streamlit App   │              │   Flask App      │        │
│  │  (app.py)        │              │  (flask_app.py)  │        │
│  │                  │              │                  │        │
│  │  • Interactive   │              │  • REST API      │        │
│  │  • Visualizations│              │  • JSON Response │        │
│  │  • Examples      │              │  • Batch Support │        │
│  └────────┬─────────┘              └────────┬─────────┘        │
│           │                                  │                   │
└───────────┼──────────────────────────────────┼───────────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                      PREDICTION LAYER                             │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              FakeNewsDetector (predict.py)                  │ │
│  │                                                             │ │
│  │  • Single Prediction      • Batch Prediction               │ │
│  │  • Confidence Scores      • Feature Extraction             │ │
│  │  • LIME Explainability    • Top Features                   │ │
│  └──────────────────┬──────────────────────────────────────────┘ │
│                     │                                             │
└─────────────────────┼─────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼──────────┐       ┌────────▼─────────┐
│  PREPROCESSING   │       │   MODEL LAYER    │
│      LAYER       │       │                  │
├──────────────────┤       ├──────────────────┤
│                  │       │                  │
│ TextPreprocessor │       │ Trained Models:  │
│ (preprocess.py)  │       │                  │
│                  │       │ • Logistic Reg.  │
│ • Text Cleaning  │       │ • Naive Bayes    │
│ • Tokenization   │       │ • DistilBERT     │
│ • Lemmatization  │       │                  │
│ • Stopword       │       │ TF-IDF           │
│   Removal        │       │ Vectorizer       │
│                  │       │                  │
└──────────────────┘       └──────────────────┘
         │                          │
         └──────────┬───────────────┘
                    │
┌───────────────────▼────────────────────────────────────────────┐
│                      TRAINING LAYER                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────┐              ┌──────────────────┐      │
│  │  Classical ML    │              │   BERT Training  │      │
│  │  (train_model.py)│              │  (train_bert.py) │      │
│  │                  │              │                  │      │
│  │  • Logistic Reg. │              │  • DistilBERT    │      │
│  │  • Naive Bayes   │              │  • Fine-tuning   │      │
│  │  • TF-IDF        │              │  • GPU Support   │      │
│  │  • Evaluation    │              │  • Transformers  │      │
│  └──────────────────┘              └──────────────────┘      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│                       DATA LAYER                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Fake.csv    │    │  True.csv    │    │  Models/     │   │
│  │  (~23K)      │    │  (~21K)      │    │  Vectorizers │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. User Interface Layer

#### Streamlit Application
- **Purpose**: Interactive web interface
- **Features**:
  - Text input area
  - File upload
  - Example selection
  - Real-time predictions
  - Interactive visualizations
  - Educational content
- **Technology**: Streamlit, Plotly, HTML/CSS

#### Flask Application
- **Purpose**: REST API and alternative web interface
- **Features**:
  - `/api/predict` - Single prediction
  - `/api/batch_predict` - Batch predictions
  - `/health` - Health check
  - HTML frontend
- **Technology**: Flask, JavaScript, HTML/CSS

### 2. Prediction Layer

#### FakeNewsDetector Class
```python
class FakeNewsDetector:
    - predict_single(text)
    - predict_batch(texts)
    - explain_prediction_lime(text)
    - get_top_features(text)
    - analyze_text(text)
```

**Responsibilities**:
- Load trained models
- Preprocess input text
- Generate predictions
- Calculate confidence scores
- Extract important features
- Provide explanations

### 3. Preprocessing Layer

#### TextPreprocessor Class
```python
class TextPreprocessor:
    - clean_text(text)
    - tokenize_and_lemmatize(text)
    - preprocess(text)
    - preprocess_dataframe(df)
```

**Pipeline**:
1. Text cleaning (lowercase, remove URLs, punctuation)
2. Tokenization (word-level)
3. Stopword removal
4. Lemmatization
5. Feature preparation

### 4. Model Layer

#### Classical ML Models
- **Logistic Regression**
  - Linear classifier
  - TF-IDF features
  - 98.5% accuracy
  - Fast inference

- **Naive Bayes**
  - Probabilistic classifier
  - TF-IDF features
  - 94.2% accuracy
  - Very fast

#### Deep Learning Model
- **DistilBERT**
  - Transformer-based
  - Pre-trained embeddings
  - ~99% accuracy
  - Slower but more accurate

#### Feature Extraction
- **TF-IDF Vectorizer**
  - 5000 max features
  - Unigrams + bigrams
  - Min/max document frequency filtering

### 5. Training Layer

#### Classical ML Training
```python
ClassicalMLModel:
    - train(X_train, y_train)
    - evaluate(X_test, y_test)
    - save_model(path)
    - load_model(path)
```

**Process**:
1. Load and preprocess data
2. Split train/test sets
3. Fit TF-IDF vectorizer
4. Train model
5. Evaluate performance
6. Save best model

#### BERT Training
```python
BERTFakeNewsDetector:
    - prepare_data(texts, labels)
    - train(train_dataset, val_dataset)
    - predict(text)
    - load_model(path)
```

**Process**:
1. Load pre-trained DistilBERT
2. Create custom dataset
3. Fine-tune on fake news data
4. Early stopping
5. Save fine-tuned model

### 6. Data Layer

#### Dataset
- **Source**: Kaggle
- **Size**: ~44,000 articles
- **Format**: CSV files
- **Labels**: FAKE / REAL

#### Model Storage
- **Format**: .pkl (pickle) for ML, .pt for BERT
- **Location**: `models/` directory
- **Components**: Model + Vectorizer

## Data Flow

### Training Flow
```
Raw Data (CSV)
    ↓
Load & Combine
    ↓
Preprocess Text
    ↓
Split Train/Test
    ↓
Feature Extraction (TF-IDF)
    ↓
Train Models
    ↓
Evaluate & Compare
    ↓
Save Best Model
```

### Prediction Flow
```
User Input (Text)
    ↓
Preprocess Text
    ↓
Vectorize (TF-IDF)
    ↓
Model Prediction
    ↓
Calculate Confidence
    ↓
Extract Features
    ↓
Generate Explanation (LIME)
    ↓
Return Results (JSON/UI)
```

## Technology Stack

### Core Libraries
- **Python**: 3.8+
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **Scikit-learn**: ML algorithms

### NLP Libraries
- **NLTK**: Text preprocessing
- **spaCy**: Advanced NLP (optional)
- **Transformers**: BERT models

### Deep Learning
- **PyTorch**: Neural networks
- **Transformers**: Pre-trained models

### Explainability
- **LIME**: Local explanations
- **SHAP**: Feature importance

### Web Frameworks
- **Streamlit**: Interactive UI
- **Flask**: REST API

### Visualization
- **Matplotlib**: Static plots
- **Seaborn**: Statistical plots
- **Plotly**: Interactive charts

## Design Patterns

### 1. Modular Design
- Separate concerns
- Independent components
- Easy to test and maintain

### 2. Factory Pattern
- Model creation
- Preprocessor initialization

### 3. Strategy Pattern
- Different model types
- Interchangeable algorithms

### 4. Singleton Pattern
- Model loading (cached)
- Resource management

## Scalability Considerations

### Current Architecture
- **Single instance**: Suitable for demo/small scale
- **In-memory models**: Fast but limited by RAM
- **Synchronous processing**: One request at a time

### Scaling Options

#### Horizontal Scaling
```
Load Balancer
    ↓
┌───────┬───────┬───────┐
│ App 1 │ App 2 │ App 3 │
└───────┴───────┴───────┘
    ↓       ↓       ↓
  Shared Model Storage
```

#### Vertical Scaling
- More RAM for larger models
- GPU for BERT inference
- Faster CPU for preprocessing

#### Optimization
- Model quantization
- Batch processing
- Caching predictions
- Async processing

## Security Considerations

### Input Validation
- Text length limits
- Character encoding checks
- Malicious input filtering

### API Security
- Rate limiting
- Authentication (optional)
- CORS configuration
- Input sanitization

### Model Security
- Model versioning
- Checksum verification
- Secure storage

## Deployment Architecture

### Development
```
Local Machine
    ↓
Python Environment
    ↓
Streamlit/Flask Server
    ↓
localhost:8501/5000
```

### Production (Cloud)
```
GitHub Repository
    ↓
CI/CD Pipeline
    ↓
Cloud Platform (Streamlit Cloud/Render/Heroku)
    ↓
HTTPS Endpoint
    ↓
Users
```

## Monitoring & Logging

### Metrics to Track
- Prediction accuracy
- Response time
- Error rate
- User engagement
- Model drift

### Logging
- Request logs
- Error logs
- Performance logs
- Model predictions

## Future Architecture Enhancements

### Microservices
```
API Gateway
    ↓
┌────────────┬────────────┬────────────┐
│ Prediction │ Training   │ Analytics  │
│ Service    │ Service    │ Service    │
└────────────┴────────────┴────────────┘
         ↓           ↓           ↓
    Database    Model Store   Metrics DB
```

### Real-time Processing
- Message queue (RabbitMQ/Kafka)
- Stream processing
- WebSocket connections

### Advanced Features
- A/B testing framework
- Model versioning system
- Automated retraining pipeline
- Multi-model ensemble

---

**This architecture provides a solid foundation for a production-ready fake news detection system while remaining simple enough for educational purposes.**
