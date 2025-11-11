"""
DistilBERT Model Training for Fake News Detection
Advanced deep learning approach using transformers
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')


class NewsDataset(Dataset):
    """
    Custom Dataset for news articles
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(pred):
    """
    Compute metrics for evaluation
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class BERTFakeNewsDetector:
    """
    DistilBERT-based fake news detector
    """
    
    def __init__(self, model_name='distilbert-base-uncased'):
        """
        Initialize BERT detector
        
        Args:
            model_name (str): Pretrained model name
        """
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def prepare_data(self, texts, labels, test_size=0.2):
        """
        Prepare data for training
        
        Args:
            texts (list): List of text samples
            labels (list): List of labels (0 for FAKE, 1 for REAL)
            test_size (float): Test set proportion
            
        Returns:
            tuple: Train and validation datasets
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = NewsDataset(X_train, y_train, self.tokenizer)
        val_dataset = NewsDataset(X_val, y_val, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, output_dir='../models/bert_model',
              epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Train the BERT model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir (str): Directory to save model
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate
        """
        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        self.model.to(self.device)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            save_total_limit=2,
            report_to='none'
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        print("\nStarting BERT model training...")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        trainer.train()
        
        # Evaluate
        print("\nEvaluating model...")
        eval_results = trainer.evaluate()
        
        print("\nEvaluation Results:")
        for key, value in eval_results.items():
            print(f"{key}: {value:.4f}")
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"\nModel saved to {output_dir}")
        
        return eval_results
    
    def predict(self, text):
        """
        Predict single text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        result = {
            'prediction': 'FAKE' if prediction == 0 else 'REAL',
            'confidence': probabilities[0][prediction].item() * 100,
            'fake_probability': probabilities[0][0].item() * 100,
            'real_probability': probabilities[0][1].item() * 100
        }
        
        return result
    
    def load_model(self, model_dir):
        """
        Load trained model
        
        Args:
            model_dir (str): Directory containing saved model
        """
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_dir}")


def train_bert_model(data_path_fake, data_path_real, output_dir='../models/bert_model',
                     epochs=3, batch_size=16):
    """
    Complete BERT training pipeline
    
    Args:
        data_path_fake (str): Path to fake news CSV
        data_path_real (str): Path to real news CSV
        output_dir (str): Directory to save model
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
    """
    from preprocess import load_and_prepare_data
    
    print("=" * 70)
    print("BERT MODEL TRAINING FOR FAKE NEWS DETECTION")
    print("=" * 70)
    
    # Load data
    df = load_and_prepare_data(data_path_fake, data_path_real)
    
    # Combine title and text
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # Convert labels to numeric (0 for FAKE, 1 for REAL)
    df['label_numeric'] = (df['label'] == 'REAL').astype(int)
    
    # Prepare data
    texts = df['combined_text'].tolist()
    labels = df['label_numeric'].tolist()
    
    # Initialize detector
    detector = BERTFakeNewsDetector()
    
    # Prepare datasets
    train_dataset, val_dataset = detector.prepare_data(texts, labels)
    
    # Train model
    results = detector.train(
        train_dataset,
        val_dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return detector, results


if __name__ == "__main__":
    fake_path = "../data/Fake.csv"
    real_path = "../data/True.csv"
    
    if os.path.exists(fake_path) and os.path.exists(real_path):
        print("Note: BERT training requires significant computational resources.")
        print("It's recommended to use a GPU for faster training.")
        print("Training may take 30-60 minutes on CPU.\n")
        
        # Uncomment to train
        # detector, results = train_bert_model(
        #     fake_path, 
        #     real_path,
        #     epochs=3,
        #     batch_size=16
        # )
        
        print("To train the BERT model, uncomment the training code in this file.")
    else:
        print("Please download the dataset and place it in the data/ directory")
