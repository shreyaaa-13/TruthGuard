"""
Prediction Module with Explainability (LIME & SHAP)
Handles inference and model interpretation
"""

import joblib
import numpy as np
import pandas as pd
from preprocess import TextPreprocessor
import warnings
warnings.filterwarnings('ignore')

# LIME for explainability
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


class FakeNewsDetector:
    """
    Complete fake news detection system with explainability
    """
    
    def __init__(self, model_path, vectorizer_path):
        """
        Initialize the detector
        
        Args:
            model_path (str): Path to trained model
            vectorizer_path (str): Path to fitted vectorizer
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.preprocessor = TextPreprocessor()
        self.class_names = ['FAKE', 'REAL']
        
        # Initialize explainers
        if LIME_AVAILABLE:
            self.lime_explainer = LimeTextExplainer(class_names=self.class_names)
        else:
            self.lime_explainer = None
            
    def predict_single(self, text):
        """
        Predict whether a single news article is fake or real
        
        Args:
            text (str): News article text
            
        Returns:
            dict: Prediction results with confidence scores
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        # Get confidence scores
        fake_confidence = probabilities[0] * 100
        real_confidence = probabilities[1] * 100
        
        result = {
            'prediction': prediction,
            'confidence': max(fake_confidence, real_confidence),
            'fake_probability': fake_confidence,
            'real_probability': real_confidence,
            'processed_text': processed_text
        }
        
        return result
    
    def predict_batch(self, texts):
        """
        Predict multiple news articles
        
        Args:
            texts (list): List of news article texts
            
        Returns:
            pd.DataFrame: Predictions with confidence scores
        """
        results = []
        
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def explain_prediction_lime(self, text, num_features=10):
        """
        Explain prediction using LIME
        
        Args:
            text (str): News article text
            num_features (int): Number of features to show
            
        Returns:
            dict: Explanation with important words
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return {"error": "LIME not available"}
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Create prediction function for LIME
        def predict_proba_fn(texts):
            processed = [self.preprocessor.preprocess(t) for t in texts]
            vectors = self.vectorizer.transform(processed)
            return self.model.predict_proba(vectors)
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            text,
            predict_proba_fn,
            num_features=num_features,
            top_labels=2
        )
        
        # Get prediction
        prediction_result = self.predict_single(text)
        
        # Extract important words for the predicted class
        predicted_class_idx = 0 if prediction_result['prediction'] == 'FAKE' else 1
        important_words = explanation.as_list(label=predicted_class_idx)
        
        result = {
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'important_words': important_words,
            'explanation_html': explanation.as_html()
        }
        
        return result
    
    def get_top_features(self, text, top_n=10):
        """
        Get top TF-IDF features for the given text
        
        Args:
            text (str): News article text
            top_n (int): Number of top features to return
            
        Returns:
            list: Top features with their scores
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Get feature names and scores
        feature_names = self.vectorizer.get_feature_names_out()
        scores = text_vector.toarray()[0]
        
        # Get top features
        top_indices = np.argsort(scores)[-top_n:][::-1]
        top_features = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        return top_features
    
    def analyze_text(self, text, explain=True):
        """
        Complete analysis of a news article
        
        Args:
            text (str): News article text
            explain (bool): Whether to include explanation
            
        Returns:
            dict: Complete analysis results
        """
        # Get prediction
        prediction_result = self.predict_single(text)
        
        # Get top features
        top_features = self.get_top_features(text, top_n=10)
        
        result = {
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'fake_probability': prediction_result['fake_probability'],
            'real_probability': prediction_result['real_probability'],
            'top_features': top_features
        }
        
        # Add explanation if requested
        if explain and LIME_AVAILABLE:
            try:
                explanation = self.explain_prediction_lime(text, num_features=10)
                result['explanation'] = explanation
            except Exception as e:
                result['explanation'] = {"error": str(e)}
        
        return result


def load_detector(model_path='../models/logistic_regression_model.pkl',
                  vectorizer_path='../models/tfidf_vectorizer.pkl'):
    """
    Load a trained fake news detector
    
    Args:
        model_path (str): Path to model file
        vectorizer_path (str): Path to vectorizer file
        
    Returns:
        FakeNewsDetector: Loaded detector
    """
    return FakeNewsDetector(model_path, vectorizer_path)


if __name__ == "__main__":
    import os
    
    # Example usage
    print("=" * 70)
    print("FAKE NEWS DETECTOR - PREDICTION MODULE")
    print("=" * 70)
    
    model_path = "../models/logistic_regression_model.pkl"
    vectorizer_path = "../models/tfidf_vectorizer.pkl"
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        # Load detector
        detector = load_detector(model_path, vectorizer_path)
        
        # Test with sample text
        sample_fake = """
        BREAKING: Scientists discover that drinking coffee can make you live forever!
        A new study shows that people who drink 10 cups of coffee per day never age.
        This shocking discovery will change everything we know about health.
        """
        
        sample_real = """
        The Federal Reserve announced today that it will maintain interest rates
        at their current levels. The decision comes after careful analysis of
        economic indicators and inflation data. Economists had widely expected
        this outcome based on recent market trends.
        """
        
        print("\n" + "=" * 70)
        print("Testing with FAKE news sample:")
        print("=" * 70)
        result1 = detector.analyze_text(sample_fake, explain=False)
        print(f"Prediction: {result1['prediction']}")
        print(f"Confidence: {result1['confidence']:.2f}%")
        print(f"Fake Probability: {result1['fake_probability']:.2f}%")
        print(f"Real Probability: {result1['real_probability']:.2f}%")
        
        print("\n" + "=" * 70)
        print("Testing with REAL news sample:")
        print("=" * 70)
        result2 = detector.analyze_text(sample_real, explain=False)
        print(f"Prediction: {result2['prediction']}")
        print(f"Confidence: {result2['confidence']:.2f}%")
        print(f"Fake Probability: {result2['fake_probability']:.2f}%")
        print(f"Real Probability: {result2['real_probability']:.2f}%")
        
    else:
        print("Model files not found. Please train the model first.")
        print(f"Expected files:")
        print(f"  - {model_path}")
        print(f"  - {vectorizer_path}")
