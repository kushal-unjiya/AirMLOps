    # /training_pipeline.py
import logging
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.data_loader import DataLoader
from src.utils.data_processor import AQIDataProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Handles the training of the air quality prediction model, combining data preparation, training,
    and evaluation steps.
    """

    def __init__(self, config_path: str = "src/config/config.json"):
        """Initialize the training pipeline with configuration."""
        self.load_config(config_path)
        self.data_loader = DataLoader()
        self.processor = AQIDataProcessor()
        self.model_path = self.config.get('model_path', 'models/model.joblib')
        self.metrics_path = os.path.join(os.path.dirname(self.model_path), 'metrics.json')
        self.required_columns = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co', 'AQI']

    def load_config(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise
    
    def load_history_data(self, city: str) -> pd.DataFrame:
        """Load and combine all historical data."""
        all_data = []
        history_dir = os.path.join("data", "historical", city)  
        
        try:
            for year in os.listdir(history_dir):
                year_dir = os.path.join(history_dir, year)
                if os.path.isdir(year_dir):
                    for month in os.listdir(year_dir):
                        month_dir = os.path.join(year_dir, month)
                        if os.path.isdir(month_dir):
                            features_path = os.path.join(month_dir, "features.csv")
                            if os.path.exists(features_path):
                                logger.info(f"Loading data from {features_path}")
                                try:
                                    df = pd.read_csv(features_path)
                                    if not df.empty:
                                        missing_columns = [col for col in self.required_columns if col not in df.columns]
                                        if missing_columns:
                                            logger.error(f"Missing columns in data: {missing_columns}")
                                            for col in missing_columns:
                                                df[col] = None
                                        all_data.append(df)
                                except Exception as e:
                                    logger.error(f"Error loading data: {str(e)}")
            if not all_data:
                logger.error("No historical data found.")
                return pd.DataFrame()
            
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded {len(combined_data)} total records from historical data.")
            return combined_data
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            return pd.DataFrame()
            
    def prepare_features(self, df: pd.DataFrame) -> list:
        """
        Prepare features and target variable for model training.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            tuple: Feature matrix (X) and target vector (y).
        """
        try:
            features_column = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
            target_column = 'AQI'
            
            missing_columns = [col for col in features_column + [target_column] if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing columns in data: {missing_columns}")
                        
            X = df[features_column]
            y = df[target_column]
            
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            logger.info(f"Prepared {len(X)} samples with {len(features_column)} features.")
            return X, y
            
                        
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def train_model(self) -> None:
        """Train the Random Forest model and save it along with evaluation metrics."""
        try:
            # Load and preprocess data
            df = self.data_loader.load_data()

            if df.empty:
                logger.error("No data available for training.")
                return

            X, y = self.prepare_features(df)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.get('test_size', 0.2), random_state=42)

            # Train the model
            model = RandomForestRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                random_state=42,
                n_jobs=-1
            )

            logger.info("Training the model...")
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

            logger.info(f"Model metrics: {metrics}")

            # Save model and metrics
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(model, self.model_path)

            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Model saved to {self.model_path}")
            logger.info(f"Metrics saved to {self.metrics_path}")

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    """Main function to run the training pipeline."""
    cities = ["London", "Beijing", "New York", "Mumbai"]  # List of cities to process

    try:
        pipeline = TrainingPipeline()
        for city in cities:
            logger.info(f"Loading historical data for {city}")
            data = pipeline.load_history_data(city)
            if data.empty:
                logger.warning(f"No data found for {city}")
            else:
                # Further processing for each city's data
                logger.info(f"Processing data for {city}")
                # Add your data processing code here
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()