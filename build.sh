#!/bin/bash
set -e

echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq build-essential swig

echo "Installing Python packages..."
pip install --upgrade pip

echo "Trying to install with FAISS..."
if pip install -r requirements.txt; then
    echo "FAISS installation successful!"
else
    echo "FAISS installation failed, trying fallback with scikit-learn..."
    pip install -r requirements-fallback.txt
    echo "Fallback installation successful!"
fi

echo "Build completed successfully!"
