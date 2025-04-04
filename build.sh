#!/usr/bin/env bash
# exit on error
set -e

echo "Starting build script..."

# Install dependencies
pip install -r requirements.txt
echo "Dependencies installed."

# Create the directory for NLTK data within the project structure
mkdir -p ./nltk_data
echo "Created ./nltk_data directory."

# Download necessary NLTK data INTO the created directory
echo "Downloading NLTK data..."
python -m nltk.downloader -d ./nltk_data stopwords wordnet omw-1.4
echo "NLTK data downloaded."

echo "Build script finished successfully."