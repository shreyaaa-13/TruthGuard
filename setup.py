"""
Setup script for Fake News Detection System
Automates initial setup and dependency installation
"""

import subprocess
import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"⏳ {description}...")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {e}")
        return False


def main():
    """Main setup function"""
    print_header("FAKE NEWS DETECTION SYSTEM - SETUP")
    
    print("This script will set up your environment for the Fake News Detection System.")
    print("It will install dependencies and download required NLTK data.\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8 or higher is recommended.")
    
    # Install dependencies
    print_header("STEP 1: Installing Dependencies")
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    ):
        print("\n⚠️  Some packages may have failed to install.")
        print("You can try installing them manually using:")
        print("pip install -r requirements.txt")
    
    # Download NLTK data
    print_header("STEP 2: Downloading NLTK Data")
    
    nltk_downloads = [
        ('punkt', 'Punkt tokenizer'),
        ('stopwords', 'Stopwords corpus'),
        ('wordnet', 'WordNet lemmatizer'),
        ('omw-1.4', 'Open Multilingual Wordnet')
    ]
    
    for package, description in nltk_downloads:
        run_command(
            f"{sys.executable} -c \"import nltk; nltk.download('{package}', quiet=True)\"",
            f"Downloading {description}"
        )
    
    # Create necessary directories
    print_header("STEP 3: Creating Directories")
    
    directories = ['data', 'models', 'notebooks', 'src', 'app', 'app/templates']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")
    
    # Final instructions
    print_header("SETUP COMPLETE!")
    
    print("Next steps:")
    print("\n1. Download the dataset:")
    print("   - Visit: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset")
    print("   - Download Fake.csv and True.csv")
    print("   - Place them in the 'data/' directory")
    
    print("\n2. Train the model:")
    print("   cd src")
    print("   python train_model.py")
    
    print("\n3. Run the web application:")
    print("   Option A (Streamlit):")
    print("   cd app")
    print("   streamlit run app.py")
    
    print("\n   Option B (Flask):")
    print("   cd app")
    print("   python flask_app.py")
    
    print("\n4. Explore the notebooks:")
    print("   jupyter notebook notebooks/EDA.ipynb")
    
    print("\n" + "=" * 70)
    print("Thank you for using the Fake News Detection System!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
