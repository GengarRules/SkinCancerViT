#!/bin/bash
# Master script to start both backend and frontend

echo "🔬 SkinCancerViT - Full Stack Launcher"
echo "======================================="
echo ""
echo "This will start both the backend (Flask) and frontend (React) servers."
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use Terminal.app
    echo "🍎 Detected macOS - Opening servers in separate Terminal windows..."
    
    # Start backend in new terminal
    osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && ./start_backend.sh\""
    
    # Wait a moment before starting frontend
    sleep 2
    
    # Start frontend in new terminal
    osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && ./start_frontend.sh\""
    
    echo ""
    echo "✅ Both servers are starting in separate Terminal windows!"
    echo ""
    echo "📍 Backend will be at: http://localhost:8000"
    echo "📍 Frontend will be at: http://localhost:5173"
    echo ""
    echo "💡 To stop the servers, close the Terminal windows or press Ctrl+C in each."
    
else
    # Linux or other Unix - use gnome-terminal or xterm
    echo "🐧 Detected Linux/Unix..."
    
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && ./start_backend.sh; exec bash"
        sleep 2
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && ./start_frontend.sh; exec bash"
        echo "✅ Both servers are starting in separate terminal windows!"
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$SCRIPT_DIR' && ./start_backend.sh" &
        sleep 2
        xterm -e "cd '$SCRIPT_DIR' && ./start_frontend.sh" &
        echo "✅ Both servers are starting in separate xterm windows!"
    else
        echo "⚠️  Could not detect terminal emulator."
        echo "Please run these commands manually in separate terminals:"
        echo ""
        echo "Terminal 1: ./start_backend.sh"
        echo "Terminal 2: ./start_frontend.sh"
    fi
fi

