import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIDataFetcher:
    """
    Handles fetching air quality data from OpenWeather API.
    """

    def __init__(self, api_key: str):
        """
        Initialize the data fetcher with API credentials.

        Args:
            api_key: OpenWeather API key
        """
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/air_pollution"

    def _make_request(self, params: dict) -> dict:
        """
        Make HTTP request to the OpenWeather API.

        Args:
            params: Query parameters for the request

        Returns:
            JSON response from the API
        """
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None

    def get_current_data(self, lat: float, lon: float) -> dict:
        """
        Fetch current air quality data for a specific location.

        Args:
            lat: Latitude of the location
            lon: Longitude of the location

        Returns:
            Dictionary containing current air quality data or None if an error occurs
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key
        }
        return self._make_request(params)

    def get_historical_data(self, lat: float, lon: float, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical air quality data for a date range.

        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            DataFrame containing historical air quality data
        """
        all_data = []
        current_date = start_date

        while current_date <= end_date:
            try:
                params = {
                    "lat": lat,
                    "lon": lon,
                    "start": int(current_date.timestamp()),
                    "end": int((current_date + timedelta(days=1)).timestamp()),
                    "appid": self.api_key
                }
                data = self._make_request(params)

                if data:
                    all_data.append(data)

                current_date += timedelta(days=1)
            except Exception as e:
                logger.error(f"Failed to fetch data for {current_date}: {str(e)}")
                current_date += timedelta(days=1)
                continue

        return pd.DataFrame(all_data)