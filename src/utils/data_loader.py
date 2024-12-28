# data_loader.py    
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and saving of data files."""
    
    def __init__(self, base_dir: str = "data"):
        """ Initialize data loader with base directory.
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.features_dir = os.path.join(base_dir, "features")
        self.historical_dir = os.path.join(base_dir, "historical")
        
        # Create directories if they don't exist
        for directory in [self.raw_dir, self.features_dir, self.historical_dir]:
            os.makedirs(directory, exist_ok=True)
            
    def save_raw_data(self, data: Dict[str, Any], city: str, timestamp: datetime) -> None:
        """
        Save raw API response data.
        
        Args:
            data: Raw API response data
            city: City name
            timestamp: Timestamp of data collection
        """
        # Create directory for city if it doesn't exist
        city_dir = os.path.join(self.raw_dir, city)
        os.makedirs(city_dir, exist_ok=True)
        
        # Save data with timestamp
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(city_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_data(self, city: str = None) -> pd.DataFrame:
        """
        Load data for a specific city or all cities if none specified.
        
        Args:
            city: Optional city name to load data for
            
        Returns:
            DataFrame containing the data 
        """
        try:
            if city:
                return self._load_city_data(city)
            else:
                return self._load_all_cities_data()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
        
    def _load_city_data(self, city: str) -> pd.DataFrame:
        city_dir = os.path.join(self.raw_dir, city)
        if not os.path.exists(city_dir):
            return pd.DataFrame()
        
        files = [f for f in os.listdir(city_dir) if f.endswith('.json')]
        if not files:
            return pd.DataFrame()
        
        dfs = []
        
        for file in files:
            with open(os.path.join(city_dir, file), 'r') as f:
                data = json.load(f)
                df = pd.DataFrame([data])
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def _load_all_cities_data(self) -> pd.DataFrame:
        all_data = []
        
        for city_dir in os.listdir(self.raw_dir):
            if os.path.isdir(os.path.join(self.raw_dir, city_dir)):
                city_df = self._load_city_data(city_dir)
                if not city_df.empty:
                    all_data.append(city_df)
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
           
    def load_features(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Load feature data within specified date range.
        
        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            DataFrame containing features
        """
        features_path = os.path.join(self.features_dir, "features.csv")
        if not os.path.exists(features_path):
            return pd.DataFrame()
            
        df = pd.read_csv(features_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
                
        return df
