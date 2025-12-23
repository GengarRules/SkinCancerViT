# 🟢 Server Status - SkinCancerViT

## ✅ Both Servers Are Running!

### Backend (Flask API)
- **Status:** 🟢 RUNNING
- **Port:** 8000
- **URL:** http://localhost:8000
- **API Endpoint:** http://localhost:8000/api/predict
- **Log File:** `/Users/pranavsreepada/mycodes/Code/skincancervt/SkinCancerViT/flask_output.log`

### Frontend (React + Vite)
- **Status:** 🟢 RUNNING  
- **Port:** 5173
- **URL:** http://localhost:5173

---

## 🌐 Access Your Application

**Open your browser and go to:**
### 👉 http://localhost:5173

---

## 📋 Quick Commands

### Check if servers are running:
```bash
# Check backend
lsof -i :8000

# Check frontend
lsof -i :5173
```

### View backend logs:
```bash
tail -f /Users/pranavsreepada/mycodes/Code/skincancervt/SkinCancerViT/flask_output.log
```

### Stop servers:
```bash
# Stop backend
pkill -f "python -m skincancer_vit.flask_api"

# Stop frontend
pkill -f "vite"
```

### Restart backend:
```bash
cd /Users/pranavsreepada/mycodes/Code/skincancervt/SkinCancerViT
source venv/bin/activate
nohup python -m skincancer_vit.flask_api > flask_output.log 2>&1 &
```

### Restart frontend:
```bash
cd /Users/pranavsreepada/mycodes/Code/skincancervt/SkinCancerViT/website
npm run dev
```

---

## 🎯 What's Installed

### Python Dependencies (Backend)
- ✅ Flask (web framework)
- ✅ flask-cors (CORS support)
- ✅ torch (PyTorch)
- ✅ transformers (HuggingFace)
- ✅ pillow (image processing)
- ✅ grad-cam (explainability)
- ✅ All other required packages

### Node Dependencies (Frontend)
- ✅ React 19
- ✅ TypeScript
- ✅ Vite (build tool)
- ✅ All required packages

---

## 📝 Notes

- The backend runs in the background and logs to `flask_output.log`
- The AI model is loaded on first request (may take a few seconds)
- Both servers will continue running until you stop them
- Virtual environment is located at: `/Users/pranavsreepada/mycodes/Code/skincancervt/SkinCancerViT/venv`

---

**Last Updated:** $(date)
**Status:** All systems operational ✅

