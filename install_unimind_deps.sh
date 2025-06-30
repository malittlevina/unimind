#!/bin/bash

# Exit on error
set -e

# Activate your virtual environment (edit the path if your venv is elsewhere)
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ venv not found! Please create one with: python3 -m venv venv"
    exit 1
fi

echo "🔍 Installing core and recommended dependencies..."

pip install --upgrade pip

pip install \
    numpy \
    opencv-python \
    scikit-learn \
    joblib \
    requests \
    openai \
    httpx \
    tqdm \
    pydantic \
    flask \
    matplotlib \
    pillow \
    pyttsx3 \
    sounddevice

echo "📦 Saving installed packages to requirements.txt..."
pip freeze > requirements.txt

echo "✅ All dependencies installed and requirements.txt updated."