#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up the Air Quality Monitoring environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
  echo "Python is not installed. Please install Python 3.9 or higher."
  exit 1
fi

# Create a virtual environment
echo "Creating a virtual environment..."
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Ensure required directories exist
echo "Setting up directories..."
mkdir -p data/features data/historical models

# Set up Streamlit configuration
echo "Configuring Streamlit..."
mkdir -p ~/.streamlit/
cat <<EOT > ~/.streamlit/config.toml
[general]
email = "your_email@example.com"
password = "your_password"
EOT

# Check for OPENWEATHERMAP_API_KEY environment variable
if [[ -z "${OPENWEATHERMAP_API_KEY}" ]]; then
  echo "OPENWEATHERMAP_API_KEY environment variable not set."
  echo "Please export your API key using: export OPENWEATHERMAP_API_KEY='your_api_key_here'"
  exit 1
fi

# Verify API access
echo "Verifying OpenWeatherMap API access..."
API_TEST_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://api.openweathermap.org/data/2.5/weather?q=London&appid=${OPENWEATHERMAP_API_KEY}")
if [[ "$API_TEST_RESPONSE" != "200" ]]; then
  echo "Failed to access OpenWeatherMap API. Please check your API key."
  exit 1
fi

echo "Setup complete. Activate the virtual environment using 'source venv/bin/activate'."
