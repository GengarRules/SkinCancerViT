#!/bin/bash
# Script to start the React frontend

echo "🚀 Starting SkinCancerViT Frontend..."
echo ""

# Navigate to the website directory
cd "$(dirname "$0")/website"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    npm install
fi

# Start the development server
echo ""
echo "✅ Starting React dev server (typically http://localhost:5173)"
echo "📝 Press Ctrl+C to stop the server"
echo ""
npm run dev

