import { useState } from 'react'
import type { ChangeEvent } from 'react'
import './App.css'

// Common body localizations for skin lesions
const LOCALIZATIONS = [
  'unknown',
  'scalp',
  'ear',
  'face',
  'neck',
  'back',
  'chest',
  'abdomen',
  'upper extremity',
  'lower extremity',
  'genital',
  'foot',
  'hand',
  'acral',
  'trunk',
  'arm',
  'leg'
]

function App() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [age, setAge] = useState<number | ''>('')
  const [localization, setLocalization] = useState<string>('unknown')
  const [prediction, setPrediction] = useState<string | null>(null)
  const [confidence, setConfidence] = useState<number | null>(null)
  const [camDataUrl, setCamDataUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files && e.target.files[0]
    if (f) {
      setImageFile(f)
      // Create preview
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result as string)
      }
      reader.readAsDataURL(f)
      // Clear previous results
      setPrediction(null)
      setConfidence(null)
      setCamDataUrl(null)
      setError(null)
    }
  }

  async function submitPrediction() {
    if (!imageFile) {
      setError('Please choose an image file first')
      return
    }
    
    setLoading(true)
    setError(null)
    
    const form = new FormData()
    form.append('image', imageFile)
    form.append('age', String(age || ''))
    form.append('localization', localization)

    try {
      const res = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data.error || 'Prediction failed')
        setPrediction(null)
        setConfidence(null)
        setCamDataUrl(null)
      } else {
        setPrediction(data.prediction)
        setConfidence(data.confidence)
        setCamDataUrl(data.cam)
        setError(null)
      }
    } catch (err) {
      console.error(err)
      setError('Error connecting to the server. Make sure the backend is running on port 8000.')
      setPrediction(null)
      setConfidence(null)
      setCamDataUrl(null)
    } finally {
      setLoading(false)
    }
  }

  function resetForm() {
    setImageFile(null)
    setImagePreview(null)
    setAge('')
    setLocalization('unknown')
    setPrediction(null)
    setConfidence(null)
    setCamDataUrl(null)
    setError(null)
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1 className="title">🔬 SkinCancerViT</h1>
          <p className="subtitle">AI-Powered Skin Lesion Analysis</p>
        </div>
      </header>

      <main className="main-content">
        <div className="content-grid">
          {/* Input Section */}
          <div className="card input-section">
            <h2 className="section-title">📤 Upload & Analyze</h2>
            
            <div className="form-group">
              <label className="form-label">Skin Lesion Image</label>
              <div className="file-upload-container">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="file-input"
                  id="file-input"
                />
                <label htmlFor="file-input" className="file-input-label">
                  {imageFile ? imageFile.name : '📁 Choose an image...'}
                </label>
              </div>
              
              {imagePreview && (
                <div className="image-preview-container">
                  <img src={imagePreview} alt="Preview" className="image-preview" />
                </div>
              )}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="age-input">
                Patient Age (optional)
              </label>
              <input
                id="age-input"
                type="number"
                min="0"
                max="120"
                value={age}
                onChange={(e) => setAge(e.target.value === '' ? '' : Number(e.target.value))}
                className="text-input"
                placeholder="Enter age..."
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="localization-select">
                Lesion Location
              </label>
              <select
                id="localization-select"
                value={localization}
                onChange={(e) => setLocalization(e.target.value)}
                className="select-input"
              >
                {LOCALIZATIONS.map((loc) => (
                  <option key={loc} value={loc}>
                    {loc.charAt(0).toUpperCase() + loc.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            <div className="button-group">
              <button
                onClick={submitPrediction}
                disabled={loading || !imageFile}
                className="btn btn-primary"
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>🔍 Analyze Lesion</>
                )}
              </button>
              
              {(imageFile || prediction) && (
                <button onClick={resetForm} className="btn btn-secondary">
                  🔄 Reset
                </button>
              )}
            </div>

            {error && (
              <div className="alert alert-error">
                <strong>⚠️ Error:</strong> {error}
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="card results-section">
            <h2 className="section-title">📊 Analysis Results</h2>
            
            {!prediction && !loading && (
              <div className="empty-state">
                <p className="empty-state-icon">🏥</p>
                <p className="empty-state-text">
                  Upload an image to see analysis results
                </p>
              </div>
            )}

            {prediction && (
              <div className="results-content">
                <div className="result-card">
                  <div className="result-header">
                    <h3 className="result-title">Diagnosis</h3>
                  </div>
                  <p className="result-value prediction-value">{prediction}</p>
                </div>

                <div className="result-card">
                  <div className="result-header">
                    <h3 className="result-title">Confidence Score</h3>
                  </div>
                  <div className="confidence-container">
                    <p className="result-value confidence-value">
                      {confidence !== null ? (confidence * 100).toFixed(2) : 0}%
                    </p>
                    <div className="confidence-bar-bg">
                      <div
                        className="confidence-bar-fill"
                        style={{ width: `${confidence !== null ? confidence * 100 : 0}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {camDataUrl && (
                  <div className="result-card cam-card">
                    <div className="result-header">
                      <h3 className="result-title">🗺️ Attention Map</h3>
                      <p className="result-subtitle">
                        Highlights areas the AI focused on
                      </p>
                    </div>
                    <div className="cam-image-container">
                      <img src={camDataUrl} alt="Class Activation Map" className="cam-image" />
                    </div>
                  </div>
                )}

                <div className="disclaimer">
                  <p>
                    <strong>⚕️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
                    Always consult with a qualified healthcare professional for medical diagnosis and treatment.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>
          Powered by Vision Transformer (ViT) | Model: <a href="https://huggingface.co/ethicalabs/SkinCancerViT" target="_blank" rel="noopener noreferrer">ethicalabs/SkinCancerViT</a>
        </p>
      </footer>
    </div>
  )
}

export default App
