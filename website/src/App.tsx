import { useState } from 'react'
import type { ChangeEvent } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [age, setAge] = useState<number | ''>('')
  const [localization, setLocalization] = useState<string>('unknown')
  const [prediction, setPrediction] = useState<string | null>(null)
  const [confidence, setConfidence] = useState<number | null>(null)
  const [camDataUrl, setCamDataUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files && e.target.files[0]
    if (f) setImageFile(f)
  }

  async function submitPrediction() {
    if (!imageFile) return alert('Please choose an image file first')
    setLoading(true)
    const form = new FormData()
    form.append('image', imageFile)
    form.append('age', String(age))
    form.append('localization', localization)

    try {
      // Adjust the URL if your Flask server is on a different host/port
      const res = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      if (!res.ok) {
        alert(data.error || 'Prediction failed')
      } else {
        setPrediction(data.prediction)
        setConfidence(data.confidence)
        setCamDataUrl(data.cam)
      }
    } catch (err) {
      console.error(err)
      alert('Error sending request to backend')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <div>
          <a href="https://vite.dev" target="_blank">
            <img src={viteLogo} className="logo" alt="Vite logo" />
          </a>
          <a href="https://react.dev" target="_blank">
            <img src={reactLogo} className="logo react" alt="React logo" />
          </a>
        </div>
        <h1>SkinCancerViT — Frontend</h1>

        <section style={{ marginTop: 20 }}>
          <div>
            <label>
              Image:&nbsp;
              <input type="file" accept="image/*" onChange={handleFileChange} />
            </label>
          </div>
          <div>
            <label>
              Age:&nbsp;
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(e.target.value === '' ? '' : Number(e.target.value))}
              />
            </label>
          </div>
          <div>
            <label>
              Localization:&nbsp;
              <input value={localization} onChange={(e) => setLocalization(e.target.value)} />
            </label>
          </div>
          <div style={{ marginTop: 10 }}>
            <button onClick={submitPrediction} disabled={loading}>
              {loading ? 'Predicting...' : 'Predict'}
            </button>
          </div>
        </section>

        <section style={{ marginTop: 20 }}>
          {prediction && (
            <div>
              <h3>Prediction</h3>
              <p>{prediction}</p>
              <p>Confidence: {confidence}</p>
            </div>
          )}
          {camDataUrl && (
            <div>
              <h3>Saliency Map (CAM)</h3>
              <img src={camDataUrl} alt="CAM" style={{ maxWidth: '320px' }} />
            </div>
          )}
        </section>
      </header>
    </div>
  )
}

export default App
