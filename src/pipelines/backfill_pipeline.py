import os
import time
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List

from src.utils.data_fetcher import AQIDataFetcher
from src.utils.data_processor import AQIDataProcessor
from src.utils.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalDataBackfill:
    """Handles backfilling of historical air quality data."""

    def __init__(self, config_path: str = "src/config/config.json"):
        """
        Initialize the backfill process.

        Args:
            config_path: Path to configuration file containing API keys and settings
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.api_key = config['api_key']
        self.cities = config.get('cities', ['beijing'])  # Default to Beijing if not specified
        self.fetcher = AQIDataFetcher(self.api_key)
        self.processor = AQIDataProcessor()
        self.loader = DataLoader()  # Uses default data directory setup

    def backfill_data(self, 
                      start_date: datetime, 
                      end_date: Optional[datetime] = None, 
                      cities: Optional[List[str]] = None) -> None:
        """
        Backfill historical data for specified cities and date range.

        Args:
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to current date)
            cities: List of cities to backfill (defaults to self.cities)
        """
        end_date = end_date or datetime.now()
        cities = cities or self.cities

        for city in cities:
            logger.info(f"Starting backfill for {city} from {start_date.date()} to {end_date.date()}")
            current_date = start_date

            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                logger.info(f"Fetching data for {city} on {date_str}")

                try:
                    # Fetch current day's data
                    raw_data = self.fetcher.get_current_data(city)

                    if raw_data:
                        # Process the data
                        processed_data = self.processor.process_data(raw_data)

                        if not processed_data.empty:
                            processed_data['date'] = date_str

                            # Save processed data to historical folder
                            year = current_date.year
                            month = current_date.month
                            os.makedirs(f"data/historical/{year}/{month:02}", exist_ok=True)
                            filepath = f"data/historical/{year}/{month:02}/features.csv"

                            # Append or create the historical file
                            if os.path.exists(filepath):
                                try:
                                    existing_df = pd.read_csv(filepath)
                                    if date_str in existing_df['date'].values:
                                        logger.info(f"Data for {date_str} already exists. Skipping.")
                                        current_date += timedelta(days=1)
                                        continue
                                except pd.errors.EmptyDataError:
                                    pass

                                processed_data = pd.concat([existing_df, processed_data], ignore_index=True)

                            processed_data.to_csv(filepath, index=False)
                            logger.info(f"Data backfilled for {date_str}.")
                        else:
                            logger.warning(f"No processable data for {date_str}")
                    else:
                        logger.error(f"Failed to fetch data for {city} on {date_str}")

                except Exception as e:
                    logger.error(f"Error processing data for {city} on {date_str}: {str(e)}")

                current_date += timedelta(days=1)
                time.sleep(1)  # Avoid overwhelming the API

    def main(self):
        """Main function to initiate the backfill process."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # Default to last 2 years

        self.backfill_data(start_date, end_date)

if __name__ == "__main__":
    backfill = HistoricalDataBackfill()
    backfill.main()
