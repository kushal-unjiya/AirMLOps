import unittest
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from src.utils import data_fetcher, data_processor
from src.pipelines import feature_pipeline
from src.utils.data_processor import AQIDataProcessor

# ------------------ Unit Tests for Utils ------------------ #

class TestDataFetcher(unittest.TestCase):
    @patch("src.utils.data_fetcher.requests.get")
    def test_get_current_data_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "data": {
                "city": "test", "aqi": 50,
                "current": {"pollution": {"aqius": 50}, "weather": {"tp": 25, "hu": 60}},
                "time": {"s": "2024-01-01 10:00:00"}
            }
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        data = data_fetcher.get_current_data("test_city")
        self.assertEqual(data["city"], "test")
        self.assertEqual(data["aqi"], 50)

    @patch("src.utils.data_fetcher.requests.get")
    def test_get_current_data_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "error", "data": {"message": "City not found"}}
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        data = data_fetcher.get_current_data("non_existent_city")
        self.assertIsNone(data)

    @patch("src.utils.data_fetcher.requests.get")
    def test_get_current_data_request_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Request error")
        data = data_fetcher.get_current_data("test_city")
        self.assertIsNone(data)

class TestDataProcessor(unittest.TestCase):
    def test_process_data_current(self):
        data = {"current": {"pollution": {"aqius": 50}, "weather": {"tp": 25, "hu": 60}}, "time": {"s": "2024-01-01 10:00:00"}}
        df = data_processor.process_data(data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df["AQI"].iloc[0], 50)
        self.assertEqual(df["temperature"].iloc[0], 25)
        self.assertEqual(df["humidity"].iloc[0], 60)

    def test_process_data_historical(self):
        data = [{"aqius": 50, "tp": 25, "hu": 60, "time": {"s": "2024-01-01 10:00:00"}},
                {"aqius": 60, "tp": 26, "hu": 65, "time": {"s": "2024-01-02 10:00:00"}}]
        df = data_processor.process_data(data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["AQI"].iloc[0], 50)
        self.assertEqual(df["AQI"].iloc[1], 60)

    def test_process_data_invalid(self):
        data = "invalid"
        df = data_processor.process_data(data)
        self.assertTrue(df.empty)

# ------------------ Unit Tests for Pipelines ------------------ #

class TestFeaturePipeline(unittest.TestCase):
    @patch("src.pipelines.feature_pipeline.data_fetcher.get_current_data")
    @patch("src.pipelines.feature_pipeline.data_processor.process_data")
    def test_main_success(self, mock_process_data, mock_get_current_data):
        mock_get_current_data.return_value = {
            "current": {"pollution": {"aqius": 50}, "weather": {"tp": 25, "hu": 60}},
            "time": {"s": "2024-01-01 10:00:00"}
        }
        mock_df = pd.DataFrame({"AQI": [50], "temperature": [25], "humidity": [60], "date": ["2024-01-01 10:00:00"]})
        mock_process_data.return_value = mock_df
        result = feature_pipeline.main("test_city")
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["features"], pd.DataFrame)
        self.assertIsInstance(result["target"], pd.Series)

    @patch("src.pipelines.feature_pipeline.data_fetcher.get_current_data")
    def test_main_failure(self, mock_get_current_data):
        mock_get_current_data.return_value = None
        result = feature_pipeline.main("test_city")
        self.assertIsNone(result)

# ------------------ Additional Tests for AQIDataProcessor ------------------ #

@pytest.mark.parametrize("sample_data,expected_aqi", [
    ({"pm2_5": 25.0, "pm10": 50.0, "o3": 30.0, "no2": 40.0, "so2": 20.0, "co": 1.0}, True)
])
def test_aqi_calculation(sample_data, expected_aqi):
    processor = AQIDataProcessor()
    row = pd.Series(sample_data)
    aqi = processor.calculate_aqi(row)
    assert isinstance(aqi, float)
    assert (aqi >= 0) == expected_aqi

def test_process_raw_data():
    processor = AQIDataProcessor()
    raw_data = {
        'iaqi': {
            'pm2_5': {'v': 25.0},
            'pm10': {'v': 50.0},
            'o3': {'v': 30.0},
            'no2': {'v': 40.0},
            'so2': {'v': 20.0},
            'co': {'v': 1.0}
        },
        'time': {'iso': '2023-12-28T12:00:00Z'}
    }
    df = processor.process_raw_data(raw_data)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'pm2_5' in df.columns
    assert 'hour' in df.columns

# ------------------ Main Test Execution ------------------ #

if __name__ == '__main__':
    unittest.main()
