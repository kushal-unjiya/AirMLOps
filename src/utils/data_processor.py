import pandas as pd
import numpy as np
from datetime import datetime
import os

class AQIDataProcessor:
    """
    Handles processing and feature engineering of air quality data.
    """

    def __init__(self):
        """
        Initialize the data processor with default parameters.
        """
        self.required_columns = [
            'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
            'temperature', 'pressure', 'humidity'
        ]

    def process_raw_data(self, data: dict) -> pd.DataFrame:
        """
        Process raw API response data into a structured format.

        Args:
            data: Raw API response dictionary

        Returns:
            DataFrame with processed features
        """
        if not data or "list" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["list"])
        df["date"] = pd.to_datetime(df["dt"], unit="s")
        df["AQI"] = df["main"].apply(lambda x: x.get("aqi"))
        df = df.select_dtypes(include=[np.number])
        return df

    def calculate_aqi(self, row: pd.Series) -> float:
        """
        Calculate Air Quality Index based on EPA standards.

        Args:
            row: Series containing pollutant measurements

        Returns:
            Calculated AQI value
        """
        weights = {
            'pm25': 0.3,
            'pm10': 0.2,
            'o3': 0.2,
            'no2': 0.1,
            'so2': 0.1,
            'co': 0.1
        }

        aqi = 0
        for pollutant, weight in weights.items():
            if pd.notna(row.get(pollutant, np.nan)):
                aqi += row[pollutant] * weight

        return round(aqi, 2)

    def save_features(self, df: pd.DataFrame, date: datetime = None) -> None:
        """
        Save processed features to appropriate location in data directory.

        Args:
            df: DataFrame containing processed features
            date: Optional date for historical data
        """
        if date:
            directory = f"data/historical/{date.year}/{date.month:02d}"
            os.makedirs(directory, exist_ok=True)
            filepath = f"{directory}/features.csv"
        else:
            directory = "data/features"
            os.makedirs(directory, exist_ok=True)
            filepath = f"{directory}/features.csv"

        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False)

    def load_historical_data(self) -> pd.DataFrame:
        """
        Load all historical data from the data directory.

        Returns:
            DataFrame containing all historical data
        """
        all_data = []
        historical_dir = "data/historical"

        for year in os.listdir(historical_dir):
            year_dir = os.path.join(historical_dir, year)
            if os.path.isdir(year_dir):
                for month in os.listdir(year_dir):
                    filepath = os.path.join(year_dir, month, "features.csv")
                    if os.path.exists(filepath):
                        df = pd.read_csv(filepath)
                        all_data.append(df)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
