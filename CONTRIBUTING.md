# Contributing to Fake News Detection System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/fake-news-detector/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Error messages and logs

### Suggesting Enhancements

1. Check existing issues and discussions
2. Create a new issue with:
   - Clear description of the enhancement
   - Use cases and benefits
   - Possible implementation approach
   - Any relevant examples

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Test your changes thoroughly
5. Commit with clear messages (`git commit -m 'Add some AmazingFeature'`)
6. Push to your branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## 📝 Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Comment complex logic

**Example:**
```python
def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text for model input.
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned and preprocessed text
    """
    # Implementation here
    pass
```

### Testing

- Write tests for new features
- Ensure existing tests pass
- Test edge cases
- Include sample data for tests

### Documentation

- Update README.md if needed
- Add docstrings to new functions
- Update QUICKSTART.md for new features
- Include examples in documentation

### Commit Messages

Use clear, descriptive commit messages:

- `feat: Add BERT model training`
- `fix: Resolve memory leak in preprocessing`
- `docs: Update installation instructions`
- `refactor: Improve code organization`
- `test: Add unit tests for predictor`

## 🎯 Areas for Contribution

### High Priority

- [ ] Multi-language support
- [ ] Real-time news feed analysis
- [ ] Improved explainability visualizations
- [ ] Mobile-responsive UI improvements
- [ ] API rate limiting and authentication

### Medium Priority

- [ ] Additional ML models (SVM, Random Forest)
- [ ] Ensemble methods
- [ ] Cross-validation improvements
- [ ] Performance optimization
- [ ] Better error handling

### Low Priority

- [ ] Dark mode for web interface
- [ ] Export results to PDF
- [ ] Browser extension
- [ ] Social media integration
- [ ] Advanced analytics dashboard

## 🔧 Setting Up Development Environment

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

4. **Download dataset and train model:**
```bash
# Follow instructions in QUICKSTART.md
```

5. **Run tests:**
```bash
pytest tests/  # If tests are available
```

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] No merge conflicts
- [ ] Changes are focused and atomic

## 🐛 Bug Fix Process

1. Create an issue describing the bug
2. Fork and create a branch: `fix/issue-number-description`
3. Fix the bug with minimal changes
4. Add tests to prevent regression
5. Update documentation if needed
6. Submit PR referencing the issue

## ✨ Feature Development Process

1. Discuss the feature in an issue first
2. Get approval from maintainers
3. Fork and create a branch: `feature/feature-name`
4. Implement the feature
5. Add comprehensive tests
6. Update documentation
7. Submit PR with detailed description

## 📚 Resources

- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [NLTK Documentation](https://www.nltk.org/)

## 🎓 Learning Resources

New to NLP or ML? Check out:

- [Natural Language Processing with Python](https://www.nltk.org/book/)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- [Fake News Detection Research Papers](https://scholar.google.com/scholar?q=fake+news+detection)

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions
- **Discussions**: General questions and ideas

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## ⚖️ Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions
- Help others learn and grow

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to fighting misinformation! 🙏**
