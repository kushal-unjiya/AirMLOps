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

    def __init__(self, config_path: str = "src/config/config.json"):
        """Initialize the feature pipeline with configuration."""
        self.load_config(config_path)
        self.fetcher = AQIDataFetcher(self.config['api_key'])
        self.processor = AQIDataProcessor()
        self.data_loader = DataLoader()

    def load_config(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    def run_pipeline(self, cities: Optional[list] = None) -> None:
        """
        Run the feature pipeline for specified cities.

        Args:
            cities: Optional list of cities to process. If None, uses cities from config.
        """
        cities = cities or self.config.get('cities', ['beijing'])
        timestamp = datetime.now()

        for city in cities:
            try:
                logger.info(f"Processing data for {city}")

                # Fetch current data
                raw_data = self.fetcher.get_current_data(city)

                if not raw_data or raw_data.get('status') != 'ok':
                    logger.error(f"Failed to fetch data for {city}: {raw_data.get('status', 'No response')}")
                    continue

                # Process data
                processed_df = self.processor.process_data(raw_data['data'])

                if processed_df.empty:
                    logger.warning(f"No processable data for {city}")
                    continue

                # Add metadata
                processed_df['city'] = city
                processed_df['timestamp'] = timestamp
                processed_df['aqi'] = processed_df.apply(self.processor.calculate_aqi, axis=1)

                # Save both raw and processed data
                self.data_loader.save_raw_data(raw_data, city, timestamp)
                self.processor.save_features(processed_df)

                logger.info(f"Successfully processed data for {city}")

            except Exception as e:
                logger.error(f"Error processing {city}: {str(e)}")
                continue

def main():
    """Main function to run the feature pipeline."""
    try:
        pipeline = FeaturePipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
