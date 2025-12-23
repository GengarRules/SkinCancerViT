#!/bin/bash
# Script to start the Flask backend

echo "🚀 Starting SkinCancerViT Backend..."
echo ""

# Navigate to the project directory
cd "$(dirname "$0")"

# Check if virtual environment exists, if not create one
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import flask" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -e .
fi

# Start the Flask server
echo ""
echo "✅ Starting Flask API on http://localhost:8000"
echo "📝 Press Ctrl+C to stop the server"
echo ""
python -m skincancer_vit.flask_api

