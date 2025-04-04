#!/usr/bin/env bash
# exit on error
set -e

echo "Starting build script..."

# Install dependencies
pip install -r requirements.txt
echo "Dependencies installed."

# --- CHANGE: Target a standard Render NLTK path ---
# Ensure the target directory exists (Render might create it, but -p is safe)
mkdir -p /opt/render/nltk_data
echo "Ensured /opt/render/nltk_data directory exists."

# Download necessary NLTK data INTO the standard Render path
echo "Downloading NLTK data (stopwords, wordnet, omw-1.4) to /opt/render/nltk_data..."
python -m nltk.downloader -d /opt/render/nltk_data stopwords wordnet omw-1.4
echo "NLTK data download attempt finished."

echo "Build script finished successfully."