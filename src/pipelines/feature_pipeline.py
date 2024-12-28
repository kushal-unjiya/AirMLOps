import logging
import sys
import os
from datetime import datetime
import json
from typing import Optional
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.data_fetcher import AQIDataFetcher
from src.utils.data_processor import AQIDataProcessor
from src.utils.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeaturePipeline:
    """Handles the feature pipeline for processing current air quality data."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
        self.load_config(self.config_path)
        self.fetcher = AQIDataFetcher(api_key)
    
    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError as e:
            logging.error(f"Failed to load config: {e}")
            raise

    def run_pipeline(self, cities: Optional[list] = None) -> None:
            """
            Run the feature pipeline for specified cities.

            Args:
                cities: Optional list of cities to process. If None, uses cities from config.
            """
            if cities is None:
                cities = self.config.get('cities', [])
            
            for city in cities:
                try:
                    logger.info(f"Processing data for {city}")
                    data = self.fetcher.fetch_data(city)  # Use the fetcher attribute
                    # Process the data...
                except Exception as e:
                    logger.error(f"Error processing {city}: {e}")

if __name__ == "__main__":
    try:
        api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
        if not api_key:
            raise ValueError("API key not provided in environment variables.")
        
        feature_pipeline = FeaturePipeline(api_key)
        feature_pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")