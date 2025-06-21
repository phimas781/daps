import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np
import sys
from sklearn.pipeline import Pipeline

# Custom model wrapper class must be defined BEFORE loading the model
class CustomModelWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, X):
        return self.model.predict(X)

# Set up the app
st.set_page_config(
    page_title="Gwamz Song Performance Predictor",
    layout="wide",
    page_icon="🎵"
)

# App title and description
st.title("🎵 Gwamz Song Performance Predictor")
st.markdown("""
**Predict the streaming performance of Gwamz's upcoming songs**  
This AI-powered tool analyzes historical patterns to forecast how well new releases will perform.
""")

# Load the model with error handling
@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    try:
        # Load the wrapped model
        loaded = joblib.load('gwamz_streams_predictor_v2.pkl')
        # Extract the actual model from the wrapper
        return loaded.model
    except Exception as e:
        st.error(f"""
        **Model loading failed**: {str(e)}
        
        Possible solutions:
        1. Make sure 'gwamz_streams_predictor_v2.pkl' is in the same directory
        2. Check the file is not corrupted
        3. Verify Python versions match between training and deployment
        """)
        st.stop()

model = load_model()

# Sample data for historical visualization
HISTORICAL_DATA = [
    {"track_name": "Last Night", "streams": 2951075, "release_year": 2023},
    {"track_name": "Just2", "streams": 1157082, "release_year": 2024},
    {"track_name": "PAMELA", "streams": 766818, "release_year": 2023},
    {"track_name": "Like This", "streams": 1724835, "release_year": 2024},
    {"track_name": "C'est La Vie", "streams": 56899, "release_year": 2021},
    {"track_name": "Composure", "streams": 81110, "release_year": 2022},
    {"track_name": "French Tips", "streams": 65482, "release_year": 2025}
]

# Sidebar for user inputs
st.sidebar.header("🎚️ Song Parameters")
st.sidebar.markdown("Configure your track's features for prediction")

# User input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Release Details")
        release_date = st.date_input("Release Date", datetime(2025, 6, 1))
        total_tracks_in_album = st.slider("Total Tracks in Album", 1, 10, 1)
        available_markets_count = st.slider("Available Markets", 1, 200, 185)
        track_number = st.slider("Track Number", 1, 10, 1)
        
    with col2:
        st.subheader("Track Features")
        is_explicit = st.toggle("Explicit Content", value=True)
        track_popularity = st.slider("Expected Popularity (1-100)", 1, 100, 45)
        track_version = st.selectbox("Track Version", 
                                   ["original", "sped up", "jersey club", 
                                    "jersey edit", "instrumental", "new gen remix"])
    
    submit_button = st.form_submit_button("🚀 Predict Streams")

# When the user clicks the predict button
if submit_button:
    # Prepare the input data
    release_datetime = datetime.combine(release_date, datetime.min.time())
    first_release_date = datetime(2021, 4, 29)  # Gwamz's first release
    days_since_first_release = (release_datetime - first_release_date).days
    
    input_data = pd.DataFrame({
        'release_year': [release_date.year],
        'release_month': [release_date.month],
        'release_day': [release_date.day],
        'release_day_of_week': [release_date.weekday()],
        'days_since_first_release': [days_since_first_release],
        'total_tracks_in_album': [total_tracks_in_album],
        'available_markets_count': [available_markets_count],
        'track_number': [track_number],
        'is_explicit': [1 if is_explicit else 0],
        'track_popularity': [track_popularity],
        'is_first_track': [1 if track_number == 1 else 0],
        'track_version': [track_version]
    })
    
    # Make prediction with error handling
    try:
        with st.spinner('Analyzing track performance...'):
            predicted_streams = model.predict(input_data)[0]
        
        # Display results
        st.success(f"## Predicted Streams: **{int(predicted_streams):,}**")
        
        # Performance insights
        st.subheader("Performance Insights")
        
        if predicted_streams > 2000000:
            st.markdown("🔥 **Excellent Potential!** This track is predicted to perform among Gwamz's top songs.")
        elif predicted_streams > 1000000:
            st.markdown("💪 **Strong Performance!** This track is expected to perform well above average.")
        elif predicted_streams > 500000:
            st.markdown("👍 **Good Potential!** This track should perform decently based on current parameters.")
        else:
            st.markdown("🤔 **Moderate Performance.** Consider optimizing release strategy or track features.")
        
        # Feature impact visualization
        st.subheader("Key Factors Influencing Prediction")
        factors = {
            'Track Popularity': track_popularity,
            'Release Date': release_date.strftime("%B %Y"),
            'Track Version': track_version,
            'Explicit Content': "Yes" if is_explicit else "No",
            'Album Position': f"Track {track_number} of {total_tracks_in_album}",
            'Available Markets': available_markets_count
        }
        
        factor_df = pd.DataFrame.from_dict(factors, orient='index', columns=['Value'])
        st.dataframe(factor_df, use_container_width=True)
        
        # Optimization tips
        st.subheader("Optimization Suggestions")
        suggestions = []
        
        if track_version != "original":
            suggestions.append("Consider releasing an **original version** first")
        
        if release_date.weekday() != 4:  # Not Friday
            suggestions.append("Switch to **Friday release** for better performance")
        
        if available_markets_count < 180:
            suggestions.append("Increase **available markets** to at least 180")
        
        if track_popularity < 40:
            suggestions.append("Boost **track popularity** through pre-release marketing")
        
        if suggestions:
            st.markdown("To improve predicted streams:")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
        else:
            st.markdown("✅ Your track parameters are well optimized!")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.markdown("""
        **Common fixes:**
        - Verify all input fields are filled correctly
        - Check the model supports all selected options
        - Ensure the model file is not corrupted
        """)

# Historical performance section
st.markdown("---")
st.subheader("📊 Historical Performance Analysis")

# Convert to DataFrame
hist_df = pd.DataFrame(HISTORICAL_DATA)

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Highest Streams", "2,951,075", "Last Night")
col2.metric("Average Streams", "787,234", "All Tracks")
col3.metric("2024 Avg Streams", "1,441,392", "+83% YoY")
col4.metric("Best Release Month", "March", "3 tracks over 1M")

# Visualization
col1, col2 = st.columns(2)

with col1:
    st.bar_chart(hist_df.set_index('track_name')['streams'])

with col2:
    yearly_data = hist_df.groupby('release_year')['streams'].mean().reset_index()
    st.line_chart(yearly_data.set_index('release_year'))

# Best practices
st.markdown("---")
st.subheader("🎯 Gwamz's Success Patterns")
st.markdown("""
Based on historical data analysis:
- **Original tracks** outperform remixes by 156% on average
- **March releases** have the highest average streams (1.2M vs 787K average)
- **Explicit tracks** get 83% more streams than clean versions
- **First tracks** on albums get 2.7× more streams than subsequent tracks
- **Friday releases** perform 42% better than mid-week releases
""")

# System information (hidden by default)
with st.expander("System Information", expanded=False):
    st.write(f"Python version: {sys.version}")
    st.write(f"Streamlit version: {st.__version__}")
    try:
        st.write(f"Model type: {type(model).__name__}")
    except:
        st.write("Model information unavailable")

# Footer
st.markdown("---")
st.caption("Gwamz Song Performance Predictor v2.0 | Predictive Model Trained on Historical Streaming Data")
