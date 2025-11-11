"""
Fake News Detection System
Source code package initialization
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .preprocess import TextPreprocessor, load_and_prepare_data
from .train_model import ClassicalMLModel
from .predict import FakeNewsDetector

__all__ = [
    'TextPreprocessor',
    'load_and_prepare_data',
    'ClassicalMLModel',
    'FakeNewsDetector'
]
