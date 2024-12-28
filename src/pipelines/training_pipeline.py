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
        self.data_loader = DataLoader(self.config['data_path'])
        self.processor = AQIDataProcessor()
        self.model_path = self.config.get('model_path', 'models/model.joblib')
        self.metrics_path = self.config.get('metrics_path', 'models/metrics.json')

    def load_config(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and target variable for model training.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            tuple: Feature matrix (X) and target vector (y).
        """
        try:
            df = self.processor.add_datetime_features(df, date_column='date')
            target_col = self.config.get('target_column', 'AQI')

            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in data.")

            feature_cols = [col for col in df.columns if col != target_col and col != 'date']

            return df[feature_cols], df[target_col]
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
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.get('test_size', 0.2), random_state=42
            )

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
    try:
        trainer = TrainingPipeline()
        trainer.train_model()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
