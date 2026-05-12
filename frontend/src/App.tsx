import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [isDragging, setIsDragging] = useState(false)
  const [status, setStatus] = useState<'idle' | 'analyzing' | 'result'>('idle')
  const [result, setResult] = useState<{verdict: string, confidence: number} | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  
  const fileInputRef = useRef<HTMLInputElement>(null)
  const terminalEndRef = useRef<HTMLDivElement>(null)

  const addLog = (msg: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`])
  }

  useEffect(() => {
    // Initial boot logs
    addLog("SYSTEM BOOT SEQUENCE INITIATED...")
    setTimeout(() => addLog("LOADING EFFICIENTNET-B0 BACKBONE: OK"), 400)
    setTimeout(() => addLog("INITIALIZING BiGRU TEMPORAL MODULE: OK"), 800)
    setTimeout(() => addLog("MOUNTING MTCNN FACE EXTRACTOR: OK"), 1200)
    setTimeout(() => addLog("AETHERIA ENGINE READY. WAITING FOR INPUT..."), 1600)
  }, [])

  useEffect(() => {
    // Auto scroll terminal
    terminalEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0])
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0])
    }
  }

  const processFile = async (file: File) => {
    if (!file.name.match(/\.(mp4|avi|mov)$/i)) {
      setError("INVALID FORMAT. REQUIRES MP4/AVI/MOV.")
      addLog(`ERROR: Invalid file format detected (${file.name})`)
      return
    }

    setError(null)
    setStatus('analyzing')
    setResult(null)
    
    addLog(`INGESTING FILE: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`)
    addLog("DISPATCHING TO FastAPI BACKEND...")
    
    const formData = new FormData()
    formData.append('video', file)

    try {
      // Simulate progressive logging for better UX
      setTimeout(() => addLog("EXTRACTING TEMPORAL SEQUENCE (16 FRAMES)..."), 1000)
      setTimeout(() => addLog("NORMALIZING VECTORS & APPLYING TRANSFORMATIONS..."), 2500)
      setTimeout(() => addLog("EXECUTING NEURAL INFERENCE..."), 4000)

      const response = await fetch('http://localhost:8000/api/detect', {
        method: 'POST',
        body: formData,
      })
      
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.detail || "ANOMALY DETECTED DURING ANALYSIS.")
      }

      let finalVerdict = data.verdict
      if (data.confidence < 60) {
        finalVerdict = "UNCERTAIN"
      }

      addLog(`ANALYSIS COMPLETE. VERDICT: ${finalVerdict} | CONF: ${data.confidence.toFixed(2)}%`)

      setResult({
        verdict: finalVerdict,
        confidence: data.confidence
      })
      setStatus('result')
      
    } catch (err: any) {
      setError(err.message || "CONNECTION FAILED.")
      addLog(`CRITICAL ERROR: ${err.message}`)
      setStatus('idle')
    }
  }

  const reset = () => {
    setStatus('idle')
    setResult(null)
    setError(null)
    addLog("SYSTEM RESET. READY FOR NEW INPUT.")
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="dashboard-wrapper">
      <nav className="top-bar">
        <div className="brand">AETHERIA // DEEPFAKE FORENSICS</div>
        <div className="status-indicator">
          <span className="pulse"></span> ONLINE
        </div>
      </nav>

      <div className="dashboard-layout">
        {/* Left Side: System Information & Terminal */}
        <aside className="side-panel">
          <div className="info-card">
            <h3>SYSTEM SPECS</h3>
            <ul>
              <li><span>MODEL:</span> EfficientNet-B0 + BiGRU</li>
              <li><span>EXTRACTOR:</span> MTCNN Cascade</li>
              <li><span>SEQ_LENGTH:</span> 16 Frames</li>
              <li><span>LATENCY:</span> ~4500ms</li>
            </ul>
          </div>

          <div className="terminal-container">
            <div className="terminal-header">
              <span>TERMINAL OUTPUT</span>
            </div>
            <div className="terminal-body">
              {logs.map((log, i) => (
                <div key={i} className="log-line">{log}</div>
              ))}
              <div ref={terminalEndRef} />
            </div>
          </div>
        </aside>

        {/* Right Side: Action Panel */}
        <main className="action-panel">
          <div className="glass-panel">
            {status === 'idle' && (
              <div 
                className={`upload-zone ${isDragging ? 'drag-active' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  style={{display: 'none'}} 
                  accept=".mp4,.avi,.mov"
                  onChange={handleFileSelect}
                />
                <span className="upload-icon">[+]</span>
                <div className="upload-text">SELECT TARGET VIDEO</div>
                <div className="upload-subtext">MP4, AVI, MOV SUPPORTED</div>
                {error && <div style={{color: 'var(--fake-color)', marginTop: '1rem', fontSize: '0.85rem'}}>{error}</div>}
              </div>
            )}

            {status === 'analyzing' && (
              <div className="loading-container">
                <div className="loading-title">ANALYSIS IN PROGRESS</div>
                <div className="loading-bar-container">
                  <div className="loading-bar"></div>
                </div>
                <div className="grid-overlay"></div>
              </div>
            )}

            {status === 'result' && result && (
              <div className="result-container">
                <div className="result-header">FINAL VERDICT</div>
                <div className={`verdict-badge verdict-${result.verdict}`}>
                  {result.verdict}
                </div>
                <div className="confidence">
                  <span>CONFIDENCE</span>
                  <span className="confidence-score">{result.confidence.toFixed(2)}%</span>
                </div>
                <button className="reset-btn" onClick={reset}>INITIATE NEW SCAN</button>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
