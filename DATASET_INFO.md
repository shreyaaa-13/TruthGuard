# Dataset Information

## Fake and Real News Dataset

### Overview

This project uses the "Fake and Real News Dataset" from Kaggle, which contains news articles labeled as either fake or real.

### Dataset Details

- **Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Creator**: Clément Bisaillon
- **License**: CC0: Public Domain

### Files

1. **Fake.csv** (~23,000 articles)
   - Contains fake news articles
   - Columns: title, text, subject, date

2. **True.csv** (~21,000 articles)
   - Contains real news articles
   - Columns: title, text, subject, date

### Data Structure

```
title: Article headline
text: Full article text
subject: News category/topic
date: Publication date
```

### Download Instructions

1. Visit the [Kaggle dataset page](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
2. Sign in to Kaggle (create account if needed)
3. Click "Download" button
4. Extract the ZIP file
5. Place `Fake.csv` and `True.csv` in the `data/` directory

### Alternative: Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset

# Extract files
unzip fake-and-real-news-dataset.zip -d data/
```

### Dataset Statistics

| Metric | Fake News | Real News | Total |
|--------|-----------|-----------|-------|
| Articles | ~23,000 | ~21,000 | ~44,000 |
| Avg. Length | ~400 words | ~450 words | ~425 words |
| Time Period | 2015-2018 | 2015-2018 | 2015-2018 |

### Topics Covered

- Politics
- World News
- Government News
- Middle East
- US News
- Left News
- Politics News

### Data Quality

**Strengths:**
- Large dataset size
- Balanced classes
- Real-world news articles
- Diverse topics

**Limitations:**
- Limited to English language
- Specific time period (2015-2018)
- May not represent current fake news patterns
- Subject labels may be biased

### Preprocessing Applied

Our system applies the following preprocessing:

1. **Text Cleaning**
   - Lowercase conversion
   - URL removal
   - Punctuation removal
   - Number removal

2. **Tokenization**
   - Word-level tokenization
   - Stopword removal

3. **Lemmatization**
   - Convert words to base form
   - Reduce vocabulary size

4. **Feature Extraction**
   - TF-IDF vectorization
   - N-gram features (1-2)

### Usage in This Project

```python
from src.preprocess import load_and_prepare_data

# Load and combine datasets
df = load_and_prepare_data(
    'data/Fake.csv',
    'data/True.csv'
)

# Dataset is automatically:
# - Combined into single DataFrame
# - Labeled (FAKE/REAL)
# - Shuffled
# - Ready for preprocessing
```

### Citation

If you use this dataset in your research, please cite:

```
Bisaillon, C. (2020). Fake and Real News Dataset. 
Kaggle. https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
```

### Ethical Considerations

- **Purpose**: Educational and research purposes only
- **Bias**: Dataset may contain inherent biases
- **Privacy**: Ensure no personal information is exposed
- **Responsibility**: Use responsibly to combat misinformation

### Additional Resources

- [Fake News Detection Research](https://scholar.google.com/scholar?q=fake+news+detection)
- [Fact-Checking Organizations](https://www.poynter.org/ifcn/)
- [Media Literacy Resources](https://medialiteracynow.org/)

### Support

For dataset-related questions:
- Visit the [Kaggle dataset page](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Check the discussion section
- Contact the dataset creator

---

**Note**: Always verify information from multiple reliable sources. This dataset is a snapshot in time and may not reflect current misinformation patterns.
