"""
Model Training Module for Fake News Detection
Supports both classical ML (Logistic Regression, Naive Bayes) and Deep Learning (DistilBERT)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import TextPreprocessor, load_and_prepare_data
import warnings
warnings.filterwarnings('ignore')


class ClassicalMLModel:
    """
    Classical Machine Learning models using TF-IDF features
    """
    
    def __init__(self, model_type='logistic_regression', max_features=5000):
        """
        Initialize the model
        
        Args:
            model_type (str): 'logistic_regression' or 'naive_bayes'
            max_features (int): Maximum number of TF-IDF features
        """
        self.model_type = model_type
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0,
                solver='liblinear'
            )
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=0.1)
        else:
            raise ValueError("model_type must be 'logistic_regression' or 'naive_bayes'")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train (list): Training texts
            y_train (list): Training labels
        """
        print(f"\nTraining {self.model_type} model...")
        
        # Fit vectorizer and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_tfidf, y_train)
        
        print("Training complete!")
    
    def predict(self, X_test):
        """
        Make predictions
        
        Args:
            X_test (list): Test texts
            
        Returns:
            tuple: (predictions, probabilities)
        """
        X_test_tfidf = self.vectorizer.transform(X_test)
        predictions = self.model.predict(X_test_tfidf)
        probabilities = self.model.predict_proba(X_test_tfidf)
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (list): Test texts
            y_test (list): True labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions, probabilities = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, pos_label='FAKE'),
            'recall': recall_score(y_test, predictions, pos_label='FAKE'),
            'f1_score': f1_score(y_test, predictions, pos_label='FAKE'),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'classification_report': classification_report(y_test, predictions)
        }
        
        return metrics
    
    def save_model(self, model_path, vectorizer_path):
        """
        Save model and vectorizer
        
        Args:
            model_path (str): Path to save model
            vectorizer_path (str): Path to save vectorizer
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path, vectorizer_path):
        """
        Load model and vectorizer
        
        Args:
            model_path (str): Path to model file
            vectorizer_path (str): Path to vectorizer file
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print("Model and vectorizer loaded successfully!")


def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm (array): Confusion matrix
        save_path (str): Path to save plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['FAKE', 'REAL'],
        yticklabels=['FAKE', 'REAL']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot comparison of different models
    
    Args:
        metrics_dict (dict): Dictionary of model metrics
        save_path (str): Path to save plot
    """
    models = list(metrics_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    data = {metric: [metrics_dict[model][metric] for model in models] for metric in metrics}
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, data[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    
    plt.close()


def train_and_evaluate_models(data_path_fake, data_path_real, models_dir='../models'):
    """
    Complete training pipeline for multiple models
    
    Args:
        data_path_fake (str): Path to fake news CSV
        data_path_real (str): Path to real news CSV
        models_dir (str): Directory to save models
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Load and preprocess data
    print("=" * 70)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("=" * 70)
    
    df = load_and_prepare_data(data_path_fake, data_path_real)
    
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)
    
    # Prepare features and labels
    X = df['processed_text'].values
    y = df['label'].values
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    models_metrics = {}
    
    # 1. Logistic Regression
    print("\n" + "=" * 70)
    print("Training Logistic Regression Model")
    print("=" * 70)
    
    lr_model = ClassicalMLModel(model_type='logistic_regression', max_features=5000)
    lr_model.train(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_test, y_test)
    models_metrics['Logistic Regression'] = lr_metrics
    
    print(f"\nLogistic Regression Results:")
    print(f"Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"Precision: {lr_metrics['precision']:.4f}")
    print(f"Recall: {lr_metrics['recall']:.4f}")
    print(f"F1-Score: {lr_metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(lr_metrics['classification_report'])
    
    # Save Logistic Regression model
    lr_model.save_model(
        os.path.join(models_dir, 'logistic_regression_model.pkl'),
        os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        lr_metrics['confusion_matrix'],
        os.path.join(models_dir, 'lr_confusion_matrix.png')
    )
    
    # 2. Naive Bayes
    print("\n" + "=" * 70)
    print("Training Naive Bayes Model")
    print("=" * 70)
    
    nb_model = ClassicalMLModel(model_type='naive_bayes', max_features=5000)
    nb_model.train(X_train, y_train)
    nb_metrics = nb_model.evaluate(X_test, y_test)
    models_metrics['Naive Bayes'] = nb_metrics
    
    print(f"\nNaive Bayes Results:")
    print(f"Accuracy: {nb_metrics['accuracy']:.4f}")
    print(f"Precision: {nb_metrics['precision']:.4f}")
    print(f"Recall: {nb_metrics['recall']:.4f}")
    print(f"F1-Score: {nb_metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(nb_metrics['classification_report'])
    
    # Save Naive Bayes model
    nb_model.save_model(
        os.path.join(models_dir, 'naive_bayes_model.pkl'),
        os.path.join(models_dir, 'nb_tfidf_vectorizer.pkl')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        nb_metrics['confusion_matrix'],
        os.path.join(models_dir, 'nb_confusion_matrix.png')
    )
    
    # Compare models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    plot_metrics_comparison(
        models_metrics,
        os.path.join(models_dir, 'models_comparison.png')
    )
    
    # Determine best model
    best_model_name = max(models_metrics, key=lambda x: models_metrics[x]['f1_score'])
    print(f"\nBest Model: {best_model_name}")
    print(f"F1-Score: {models_metrics[best_model_name]['f1_score']:.4f}")
    
    return models_metrics


if __name__ == "__main__":
    # Example usage
    fake_path = "../data/Fake.csv"
    real_path = "../data/True.csv"
    
    if os.path.exists(fake_path) and os.path.exists(real_path):
        train_and_evaluate_models(fake_path, real_path)
    else:
        print("Please download the dataset and place it in the data/ directory")
        print("Dataset: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset")
