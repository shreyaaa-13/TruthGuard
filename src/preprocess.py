"""
Text Preprocessing Module for Fake News Detection
Handles cleaning, tokenization, lemmatization, and vectorization
"""

import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for news articles
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Clean and preprocess a single text string
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize text and apply lemmatization
        
        Args:
            text (str): Cleaned text
            
        Returns:
            list: List of lemmatized tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Preprocessed text ready for vectorization
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='text', title_column='title'):
        """
        Preprocess entire dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            title_column (str): Name of title column
            
        Returns:
            pd.DataFrame: Dataframe with preprocessed text
        """
        print("Starting text preprocessing...")
        
        # Combine title and text
        df['combined_text'] = df[title_column].fillna('') + ' ' + df[text_column].fillna('')
        
        # Apply preprocessing
        print("Cleaning and preprocessing text...")
        df['processed_text'] = df['combined_text'].apply(self.preprocess)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
        
        print(f"Preprocessing complete! Processed {len(df)} articles.")
        
        return df


def load_and_prepare_data(fake_path, real_path):
    """
    Load fake and real news datasets and combine them
    
    Args:
        fake_path (str): Path to fake news CSV
        real_path (str): Path to real news CSV
        
    Returns:
        pd.DataFrame: Combined and labeled dataset
    """
    print("Loading datasets...")
    
    # Load datasets
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)
    
    # Add labels
    fake_df['label'] = 'FAKE'
    real_df['label'] = 'REAL'
    
    # Combine datasets
    df = pd.concat([fake_df, real_df], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Loaded {len(fake_df)} fake articles and {len(real_df)} real articles")
    print(f"Total: {len(df)} articles")
    
    return df


def get_data_statistics(df):
    """
    Get basic statistics about the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'total_articles': len(df),
        'fake_count': len(df[df['label'] == 'FAKE']),
        'real_count': len(df[df['label'] == 'REAL']),
        'avg_text_length': df['text'].str.len().mean(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Text Preprocessor Module")
    print("=" * 50)
    
    # Test preprocessing
    preprocessor = TextPreprocessor()
    
    sample_text = """
    Breaking News: This is a SAMPLE article with URLs http://example.com 
    and numbers 12345. It contains punctuation!!! And @mentions #hashtags.
    """
    
    cleaned = preprocessor.preprocess(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"\nCleaned: {cleaned}")
