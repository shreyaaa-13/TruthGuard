"""
Flask Web Application for Fake News Detection
Alternative to Streamlit with REST API support
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import FakeNewsDetector
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'logistic_regression_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')

detector = None

def load_model():
    """Load the trained model"""
    global detector
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        detector = FakeNewsDetector(MODEL_PATH, VECTORIZER_PATH)
        return True
    return False


@app.route('/')
def home():
    """Home page"""
    model_loaded = detector is not None
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    if detector is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({
                'error': 'No text provided'
            }), 400
        
        # Get prediction
        result = detector.analyze_text(text, explain=False)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 2),
            'fake_probability': round(result['fake_probability'], 2),
            'real_probability': round(result['real_probability'], 2),
            'top_features': result['top_features'][:10]
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch prediction"""
    if detector is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({
                'error': 'No texts provided'
            }), 400
        
        # Get predictions
        results = []
        for text in texts:
            result = detector.predict_single(text)
            results.append({
                'prediction': result['prediction'],
                'confidence': round(result['confidence'], 2),
                'fake_probability': round(result['fake_probability'], 2),
                'real_probability': round(result['real_probability'], 2)
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None
    })


if __name__ == '__main__':
    print("=" * 70)
    print("FAKE NEWS DETECTOR - FLASK APPLICATION")
    print("=" * 70)
    
    # Load model
    if load_model():
        print("✓ Model loaded successfully!")
    else:
        print("⚠ Warning: Model not found. Please train the model first.")
        print("Run: python src/train_model.py")
    
    print("\nStarting Flask server...")
    print("Access the app at: http://localhost:5000")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
