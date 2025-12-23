# 🔬 SkinCancerViT Web Application Setup Guide

This guide will help you set up and run the full-stack web application for skin cancer lesion analysis using AI.

## 📋 Prerequisites

- **Python** 3.10.17 or higher
- **Node.js** 16+ and npm (for the frontend)
- **Git** (for cloning and managing the repository)

## 🚀 Quick Start

### Step 1: Backend Setup (Flask API)

1. **Navigate to the project root:**
   ```bash
   cd /Users/pranavsreepada/mycodes/Code/skincancervt/SkinCancerViT
   ```

2. **Install Python dependencies using uv (recommended) or pip:**

   **Option A: Using uv (faster, recommended)**
   ```bash
   # Install uv if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install dependencies
   uv pip install -e .
   ```

   **Option B: Using pip**
   ```bash
   # Create a virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -e .
   ```

3. **Start the Flask backend:**
   ```bash
   python -m skincancer_vit.flask_api
   ```

   The backend will start on `http://localhost:8000`

   **Expected output:**
   ```
   Loading model ethicalabs/SkinCancerViT to cpu...
   Model loaded.
    * Running on all addresses (0.0.0.0)
    * Running on http://127.0.0.1:8000
   ```

### Step 2: Frontend Setup (React + Vite)

1. **Open a new terminal** and navigate to the website folder:
   ```bash
   cd /Users/pranavsreepada/mycodes/Code/skincancervt/SkinCancerViT/website
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

   The frontend will start on `http://localhost:5173` (or another port if 5173 is busy)

   **Expected output:**
   ```
   VITE v5.x.x  ready in xxx ms
   
   ➜  Local:   http://localhost:5173/
   ➜  Network: use --host to expose
   ```

4. **Open your browser** and navigate to `http://localhost:5173`

## 🎯 Using the Application

1. **Upload an image** of a skin lesion
2. **Enter patient information:**
   - Age (optional)
   - Lesion location on the body
3. **Click "Analyze Lesion"** to get predictions
4. **View results:**
   - Diagnosis prediction
   - Confidence score
   - Attention map (showing which parts of the image the AI focused on)

## 🏗️ Project Structure

```
SkinCancerViT/
├── skincancer_vit/          # Python backend package
│   ├── flask_api.py         # Flask API server
│   ├── model.py             # AI model implementation
│   ├── xai_utils.py         # Explainable AI utilities (CAM)
│   └── ...
├── website/                 # React frontend
│   ├── src/
│   │   ├── App.tsx          # Main application component
│   │   ├── App.css          # Application styles
│   │   └── ...
│   └── package.json
├── pyproject.toml           # Python dependencies
└── WEBAPP_SETUP.md          # This file
```

## 🔧 Configuration

### Backend Configuration

The backend uses:
- **Model:** `ethicalabs/SkinCancerViT` from HuggingFace
- **Port:** 8000
- **CORS:** Enabled for `http://localhost:5173`

To change the port, modify `flask_api.py`:
```python
app.run(host="0.0.0.0", port=YOUR_PORT, debug=True)
```

### Frontend Configuration

To change the backend API URL, modify `App.tsx`:
```typescript
const res = await fetch('http://YOUR_BACKEND_URL/api/predict', {
```

## 📦 API Endpoints

### POST `/api/predict`

Analyzes a skin lesion image and returns predictions.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Fields:
  - `image`: Image file (required)
  - `age`: Patient age as integer (optional)
  - `localization`: Body location string (required)

**Response:**
```json
{
  "prediction": "melanoma",
  "confidence": 0.87,
  "cam": "data:image/png;base64,..."
}
```

## 🐛 Troubleshooting

### Backend Issues

**Problem:** Model fails to load
- **Solution:** Ensure you have a stable internet connection. The model will be downloaded from HuggingFace on first run.

**Problem:** `ModuleNotFoundError: No module named 'flask'`
- **Solution:** Make sure you installed all dependencies with `uv pip install -e .` or `pip install -e .`

**Problem:** Port 8000 already in use
- **Solution:** Change the port in `flask_api.py` or stop the process using port 8000

### Frontend Issues

**Problem:** `fetch` error or CORS issues
- **Solution:** Make sure the Flask backend is running on port 8000

**Problem:** `Cannot find module` errors
- **Solution:** Run `npm install` again in the website directory

**Problem:** Port 5173 already in use
- **Solution:** Vite will automatically use a different port. Check the terminal output for the actual URL.

## 📱 Building for Production

### Backend

For production, use a WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 skincancer_vit.flask_api:app
```

### Frontend

Build the optimized production bundle:
```bash
cd website
npm run build
```

The built files will be in `website/dist/`. Serve them with any static file server.

## ⚕️ Medical Disclaimer

**IMPORTANT:** This application is designed for research and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the terminal output for error messages
3. Check that both frontend and backend are running
4. Verify all dependencies are installed correctly

## 📄 License

This project uses the SkinCancerViT model from HuggingFace. Please refer to the model card for license information:
- https://huggingface.co/ethicalabs/SkinCancerViT

---

**Developed with ❤️ using React, TypeScript, Flask, and PyTorch**

