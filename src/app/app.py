import streamlit as st
import joblib
import os
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.data_fetcher import AQIDataFetcher
from src.utils.data_processor import AQIDataProcessor
from src.pipelines.feature_pipeline import FeaturePipeline

def load_model():
    """Load the trained model."""
    model_path = 'models/model.joblib'
    if not os.path.exists(model_path):
        st.error("Model not found. Please train the model first.")
        return None
    return joblib.load(model_path)

def main():
    st.title("Air Quality Prediction Dashboard")

    # Load configuration
    try:
        with open("src/config/config.json", 'r') as f:
            config = json.load(f)
    except Exception as e:
        st.error("Failed to load configuration. Please check config.json")
        return

    # Initialize components
    api_key = config.get('api_key', os.environ.get("OPENWEATHERMAP_API_KEY"))
    if not api_key:
        st.error("API key not provided in config or environment.")
        return

    model = load_model()
    fetcher = AQIDataFetcher(api_key)
    processor = AQIDataProcessor()
    feature_pipeline = FeaturePipeline(api_key)

    # City input
    city_name = st.text_input("Enter City Name", "London")

    if st.button("Get Prediction"):
        try:
            # Fetch and process data
            processed_data = feature_pipeline.run_pipeline[(city_name)]
            if processed_data.empty:
                st.error(f"Could not retrieve or process data for {[city_name]}. Please check the city name and API availability.")
                return

            # processed_data['month'] = pd.to_datetime(processed_data['date']).dt.month
            # processed_data['day'] = pd.to_datetime(processed_data['date']).dt.day
            # processed_data['year'] = pd.to_datetime(processed_data['date']).dt.year
            # processed_data = processed_data.drop('date', axis=1)

            # Make predictions
            # if model is not None:
            prediction = model.predict(processed_data)

            st.subheader(f"Predicted AQI for {[city_name]}")
            st.metric("Predicted AQI", f"{prediction:.2f}")

            # Pollutant levels visualization
            st.subheader("Processed Pollutant Levels")
            fig = go.Figure()
            pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
            values = [processed_data.iloc[0][p] for p in pollutants]

            fig.add_trace(go.Bar(
                x=pollutants,
                y=values,
                text=values,
                textposition='auto',
            ))

            fig.update_layout(
                title="Pollutant Concentrations",
                xaxis_title="Pollutant",
                yaxis_title="Concentration"
            )
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
