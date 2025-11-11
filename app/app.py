"""
Streamlit Web Application for Fake News Detection
Beautiful, modern UI with explainability features
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import FakeNewsDetector
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .fake-news {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .real-news {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .confidence-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .feature-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached)"""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'logistic_regression_model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        return FakeNewsDetector(model_path, vectorizer_path)
    else:
        return None


def create_confidence_gauge(confidence, prediction):
    """Create a gauge chart for confidence score"""
    color = "#4caf50" if prediction == "REAL" else "#f44336"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_probability_chart(fake_prob, real_prob):
    """Create a bar chart for probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Fake', 'Real'],
            y=[fake_prob, real_prob],
            marker_color=['#f44336', '#4caf50'],
            text=[f'{fake_prob:.1f}%', f'{real_prob:.1f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Category",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_feature_chart(features):
    """Create a horizontal bar chart for top features"""
    if not features:
        return None
    
    words = [f[0] for f in features]
    scores = [f[1] for f in features]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=words,
        orientation='h',
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title="Top Important Keywords (TF-IDF Scores)",
        xaxis_title="TF-IDF Score",
        yaxis_title="Keywords",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🔍 Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered News Verification System using NLP & Machine Learning</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This system uses Natural Language Processing and Machine Learning 
        to detect whether a news article is fake or real.
        """)
        
        st.header("🎯 Features")
        st.write("""
        - **Text Analysis**: Advanced NLP preprocessing
        - **ML Models**: Logistic Regression with TF-IDF
        - **Explainability**: See which keywords influenced the decision
        - **Confidence Scores**: Know how certain the model is
        """)
        
        st.header("📊 Model Info")
        st.write("""
        - **Algorithm**: Logistic Regression
        - **Features**: TF-IDF (5000 features)
        - **Training Data**: 40,000+ news articles
        - **Accuracy**: ~98%
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for educational purposes. Always verify news from 
        multiple reliable sources before drawing conclusions.
        """)
    
    # Load model
    detector = load_model()
    
    if detector is None:
        st.error("⚠️ Model not found! Please train the model first by running `python src/train_model.py`")
        st.info("""
        **Steps to get started:**
        1. Download the dataset from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
        2. Place `Fake.csv` and `True.csv` in the `data/` directory
        3. Run `python src/train_model.py` to train the model
        4. Restart this app
        """)
        return
    
    # Main content
    st.header("📝 Enter News Article")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Paste Text", "Upload File", "Try Examples"],
        horizontal=True
    )
    
    text_input = ""
    
    if input_method == "Paste Text":
        text_input = st.text_area(
            "Paste the news article text here:",
            height=200,
            placeholder="Enter or paste a news article here..."
        )
    
    elif input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a text file (.txt)",
            type=['txt']
        )
        if uploaded_file is not None:
            text_input = uploaded_file.read().decode('utf-8')
            st.text_area("File content:", text_input, height=200)
    
    elif input_method == "Try Examples":
        example_choice = st.selectbox(
            "Select an example:",
            [
                "Select an example...",
                "Example 1: Fake News (Sensational)",
                "Example 2: Real News (Political)",
                "Example 3: Fake News (Health Misinformation)"
            ]
        )
        
        examples = {
            "Example 1: Fake News (Sensational)": """
                BREAKING: Scientists Discover That Eating Chocolate Makes You Immortal!
                A groundbreaking study reveals that consuming 10 chocolate bars daily can 
                completely stop the aging process. Researchers claim this miracle food 
                contains secret compounds that reverse aging. Doctors are shocked by these 
                findings that pharmaceutical companies don't want you to know about!
            """,
            "Example 2: Real News (Political)": """
                The Federal Reserve announced today that it will maintain interest rates 
                at their current levels following a two-day policy meeting. The decision 
                comes after careful analysis of economic indicators including inflation 
                data and employment figures. Fed Chair stated that the committee will 
                continue to monitor economic conditions and adjust policy as needed to 
                support maximum employment and price stability.
            """,
            "Example 3: Fake News (Health Misinformation)": """
                URGENT: New Vaccine Contains Microchips to Track Your Every Move!
                Leaked documents reveal that the government is using vaccines to implant 
                tracking devices in citizens. Anonymous sources confirm that these chips 
                can control your thoughts and monitor your location 24/7. Share this 
                before it gets deleted! The mainstream media won't report this truth!
            """
        }
        
        if example_choice != "Select an example...":
            text_input = examples[example_choice]
            st.text_area("Example text:", text_input, height=200)
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Article", use_container_width=True)
    
    # Analysis
    if analyze_button and text_input.strip():
        with st.spinner("🔄 Analyzing article..."):
            try:
                # Get prediction
                result = detector.analyze_text(text_input, explain=False)
                
                prediction = result['prediction']
                confidence = result['confidence']
                fake_prob = result['fake_probability']
                real_prob = result['real_probability']
                top_features = result['top_features']
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Prediction result
                if prediction == "FAKE":
                    st.markdown(f"""
                    <div class="fake-news">
                        <h2 style="color: #f44336; margin: 0;">❌ FAKE NEWS DETECTED</h2>
                        <p style="font-size: 1.1rem; margin-top: 10px;">
                            This article shows characteristics of fake or misleading news.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="real-news">
                        <h2 style="color: #4caf50; margin: 0;">✅ LIKELY REAL NEWS</h2>
                        <p style="font-size: 1.1rem; margin-top: 10px;">
                            This article appears to be legitimate news content.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confidence gauge
                    gauge_fig = create_confidence_gauge(confidence, prediction)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with col2:
                    # Probability chart
                    prob_fig = create_probability_chart(fake_prob, real_prob)
                    st.plotly_chart(prob_fig, use_container_width=True)
                
                # Top features
                st.subheader("🔑 Key Factors in Decision")
                
                if top_features:
                    feature_fig = create_feature_chart(top_features)
                    st.plotly_chart(feature_fig, use_container_width=True)
                    
                    st.info("""
                    **How to interpret:** The chart above shows the most important keywords 
                    from the article based on TF-IDF scores. These words had the strongest 
                    influence on the model's decision.
                    """)
                else:
                    st.warning("No significant features found in the text.")
                
                # Additional insights
                with st.expander("📈 Detailed Metrics"):
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.metric("Prediction", prediction)
                        st.metric("Fake Probability", f"{fake_prob:.2f}%")
                    
                    with metrics_col2:
                        st.metric("Confidence", f"{confidence:.2f}%")
                        st.metric("Real Probability", f"{real_prob:.2f}%")
                
                # Tips
                with st.expander("💡 Tips for Identifying Fake News"):
                    st.markdown("""
                    - **Check the source**: Is it from a reputable news organization?
                    - **Look for evidence**: Are there credible sources and citations?
                    - **Check the date**: Is the information current and relevant?
                    - **Verify with other sources**: Do other outlets report the same story?
                    - **Watch for bias**: Is the language emotional or sensational?
                    - **Check the author**: Is the author credible and identifiable?
                    - **Be skeptical**: If it sounds too good (or bad) to be true, verify it!
                    """)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.exception(e)
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Fake News Detection System</strong> | Built with Python, Scikit-learn, and Streamlit</p>
        <p>⚠️ <em>This tool is for educational purposes. Always verify information from multiple reliable sources.</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
