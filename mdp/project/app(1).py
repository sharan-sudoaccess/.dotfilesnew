# ============================================================
# Flask Web App for Metal Removal Efficiency Predictor
# Run: python3 app.py   then open http://localhost:5000
# ============================================================
from flask import Flask, request, jsonify, render_template_string
import numpy as np, joblib, os, sys

app = Flask(__name__)

if not os.path.exists('model.pkl'):
    print("ERROR: model.pkl not found. Run ANN_Metal_Removal.py first.")
    sys.exit(1)

data     = joblib.load('model.pkl')
net      = data['net']
scaler_X = data['scaler_X']
scaler_Y = data['scaler_Y']

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Metal Removal Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0f0a;
    --surface: #111811;
    --card: #161e16;
    --border: #2a3a2a;
    --accent: #4ade80;
    --accent2: #22c55e;
    --accent-dim: rgba(74,222,128,0.12);
    --text: #e8f5e8;
    --text-dim: #6b8f6b;
    --danger: #f87171;
    --warning: #fbbf24;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Mono', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* animated grid background */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(74,222,128,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(74,222,128,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
  }

  .glow-orb {
    position: fixed;
    width: 500px; height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(74,222,128,0.06) 0%, transparent 70%);
    top: -100px; left: -100px;
    pointer-events: none; z-index: 0;
    animation: drift 8s ease-in-out infinite alternate;
  }
  @keyframes drift {
    to { transform: translate(60px, 80px); }
  }

  .wrapper {
    position: relative; z-index: 1;
    max-width: 860px;
    margin: 0 auto;
    padding: 48px 24px;
  }

  /* Header */
  header { margin-bottom: 40px; }
  .badge {
    display: inline-block;
    font-size: 10px; font-weight: 500; letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 4px 10px; border-radius: 2px;
    margin-bottom: 16px;
  }
  h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(28px, 5vw, 44px);
    font-weight: 800; line-height: 1.1;
    color: var(--text);
    margin-bottom: 10px;
  }
  h1 span { color: var(--accent); }
  .subtitle {
    color: var(--text-dim);
    font-size: 13px; line-height: 1.6;
  }

  /* Metal selector */
  .metal-tabs {
    display: flex; gap: 8px;
    margin-bottom: 28px;
  }
  .metal-tab {
    flex: 1; padding: 12px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--text-dim);
    font-family: 'Syne', sans-serif;
    font-size: 14px; font-weight: 600;
    cursor: pointer; border-radius: 6px;
    transition: all 0.2s;
    text-align: center;
  }
  .metal-tab:hover { border-color: var(--accent); color: var(--accent); }
  .metal-tab.active {
    background: var(--accent-dim);
    border-color: var(--accent);
    color: var(--accent);
  }

  /* Grid of inputs */
  .inputs-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }
  @media (max-width: 520px) { .inputs-grid { grid-template-columns: 1fr; } }

  .field {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    transition: border-color 0.2s;
  }
  .field:focus-within { border-color: var(--accent); }

  .field label {
    display: flex; justify-content: space-between;
    font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--text-dim); margin-bottom: 10px;
  }
  .field label .unit {
    color: var(--accent); font-size: 10px;
  }

  .slider-row {
    display: flex; align-items: center; gap: 12px;
  }
  input[type=range] {
    flex: 1; height: 3px;
    -webkit-appearance: none;
    background: var(--border); border-radius: 2px; outline: none;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--accent); cursor: pointer;
    box-shadow: 0 0 8px rgba(74,222,128,0.5);
  }
  .val-display {
    font-family: 'Syne', sans-serif;
    font-size: 18px; font-weight: 700;
    color: var(--accent); min-width: 48px; text-align: right;
  }

  /* Predict button */
  .btn-predict {
    width: 100%; padding: 16px;
    background: var(--accent);
    color: #0a0f0a;
    font-family: 'Syne', sans-serif;
    font-size: 15px; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    border: none; border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    position: relative; overflow: hidden;
  }
  .btn-predict:hover {
    background: var(--accent2);
    box-shadow: 0 0 24px rgba(74,222,128,0.35);
    transform: translateY(-1px);
  }
  .btn-predict:active { transform: translateY(0); }
  .btn-predict.loading { opacity: 0.7; cursor: not-allowed; }

  /* Result card */
  .result-card {
    margin-top: 24px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 28px;
    display: none;
    animation: slideUp 0.3s ease;
  }
  .result-card.show { display: block; }
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .result-label {
    font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
    color: var(--text-dim); margin-bottom: 8px;
  }
  .result-value {
    font-family: 'Syne', sans-serif;
    font-size: 56px; font-weight: 800;
    line-height: 1; margin-bottom: 4px;
  }
  .result-value.excellent { color: var(--accent); }
  .result-value.good      { color: var(--warning); }
  .result-value.poor      { color: var(--danger); }

  .result-meta {
    font-size: 12px; color: var(--text-dim); margin-bottom: 20px;
  }

  /* Progress bar */
  .bar-wrap {
    background: var(--border); border-radius: 2px;
    height: 6px; margin-bottom: 20px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    transition: width 0.8s cubic-bezier(.23,1,.32,1);
    width: 0%;
  }

  /* Stats row */
  .stats-row {
    display: grid; grid-template-columns: repeat(3,1fr); gap: 12px;
  }
  .stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px; padding: 12px;
    text-align: center;
  }
  .stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 18px; font-weight: 700; color: var(--text);
  }
  .stat-lbl {
    font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--text-dim); margin-top: 4px;
  }

  /* Footer note */
  .note {
    margin-top: 32px; padding: 14px 16px;
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--surface);
    font-size: 11px; color: var(--text-dim); line-height: 1.6;
  }
  .note strong { color: var(--accent); }
</style>
</head>
<body>
<div class="glow-orb"></div>
<div class="wrapper">

  <header>
    <div class="badge">ANN Model · R² = 0.9673</div>
    <h1>Metal Removal<br><span>Efficiency</span> Predictor</h1>
    <p class="subtitle">
      Artificial Neural Network trained on Cu(II) &amp; Cr(VI) adsorption data<br>
      from Muraya koenigii biosorbent &amp; Hematite nanoparticles studies.
    </p>
  </header>

  <!-- Metal selector -->
  <div class="metal-tabs">
    <button class="metal-tab active" onclick="selectMetal('Cu', this)">
      Cu(II) — Copper
    </button>
    <button class="metal-tab" onclick="selectMetal('Cr', this)">
      Cr(VI) — Chromium
    </button>
  </div>
  <input type="hidden" id="metal" value="Cu">

  <!-- Sliders -->
  <div class="inputs-grid">

    <div class="field">
      <label>pH <span class="unit">2 – 7</span></label>
      <div class="slider-row">
        <input type="range" id="ph" min="2" max="7" step="0.1" value="6"
               oninput="update('ph','phVal')">
        <span class="val-display" id="phVal">6.0</span>
      </div>
    </div>

    <div class="field">
      <label>Temperature <span class="unit">°C</span></label>
      <div class="slider-row">
        <input type="range" id="temp" min="25" max="65" step="1" value="35"
               oninput="update('temp','tempVal')">
        <span class="val-display" id="tempVal">35</span>
      </div>
    </div>

    <div class="field">
      <label>Contact Time <span class="unit">min</span></label>
      <div class="slider-row">
        <input type="range" id="time" min="5" max="120" step="5" value="30"
               oninput="update('time','timeVal')">
        <span class="val-display" id="timeVal">30</span>
      </div>
    </div>

    <div class="field">
      <label>Concentration <span class="unit">mg/L</span></label>
      <div class="slider-row">
        <input type="range" id="conc" min="10" max="100" step="5" value="20"
               oninput="update('conc','concVal')">
        <span class="val-display" id="concVal">20</span>
      </div>
    </div>

    <div class="field">
      <label>Adsorbent Dosage <span class="unit">g/L</span></label>
      <div class="slider-row">
        <input type="range" id="dosage" min="0.1" max="2.0" step="0.1" value="0.5"
               oninput="update('dosage','dosageVal')">
        <span class="val-display" id="dosageVal">0.5</span>
      </div>
    </div>

    <div class="field">
      <label>Model Info <span class="unit">ANN</span></label>
      <div style="padding-top:6px; font-size:11px; color:var(--text-dim); line-height:1.8;">
        Architecture: 6–8–1<br>
        Algorithm: L-BFGS<br>
        Training R²: 0.9862
      </div>
    </div>

  </div>

  <button class="btn-predict" id="predictBtn" onclick="predict()">
    ⚗ Predict Removal Efficiency
  </button>

  <!-- Result -->
  <div class="result-card" id="resultCard">
    <div class="result-label">Predicted Removal Efficiency</div>
    <div class="result-value" id="resultValue">—</div>
    <div class="result-meta" id="resultMeta"></div>
    <div class="bar-wrap"><div class="bar-fill" id="barFill"></div></div>
    <div class="stats-row">
      <div class="stat">
        <div class="stat-val" id="s1">±2.95%</div>
        <div class="stat-lbl">MAE Error</div>
      </div>
      <div class="stat">
        <div class="stat-val" id="s2">±4.99%</div>
        <div class="stat-lbl">RMSE Error</div>
      </div>
      <div class="stat">
        <div class="stat-val">0.9673</div>
        <div class="stat-lbl">Overall R²</div>
      </div>
    </div>
  </div>

  <div class="note">
    <strong>Model:</strong> ANN trained on 140 experimental data points from two research papers.
    Inputs: pH, Temperature, Contact Time, Concentration, Dosage, Metal type.
    Predicted value carries an uncertainty of ±2.95% (MAE).
  </div>

</div>

<script>
  function update(id, displayId) {
    const val = parseFloat(document.getElementById(id).value);
    document.getElementById(displayId).textContent =
      id === 'dosage' ? val.toFixed(1) :
      id === 'ph'     ? val.toFixed(1) : val;
  }

  function selectMetal(m, el) {
    document.querySelectorAll('.metal-tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('metal').value = m;
  }

  async function predict() {
    const btn = document.getElementById('predictBtn');
    btn.classList.add('loading');
    btn.textContent = 'Predicting...';

    const payload = {
      metal:  document.getElementById('metal').value,
      ph:     parseFloat(document.getElementById('ph').value),
      temp:   parseFloat(document.getElementById('temp').value),
      time:   parseFloat(document.getElementById('time').value),
      conc:   parseFloat(document.getElementById('conc').value),
      dosage: parseFloat(document.getElementById('dosage').value),
    };

    try {
      const res  = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      const card = document.getElementById('resultCard');
      const valEl = document.getElementById('resultValue');
      const v = data.removal;

      valEl.textContent = v.toFixed(2) + '%';
      valEl.className = 'result-value ' +
        (v >= 85 ? 'excellent' : v >= 60 ? 'good' : 'poor');

      document.getElementById('resultMeta').textContent =
        `${payload.metal === 'Cu' ? 'Cu(II)' : 'Cr(VI)'} removal · ` +
        `Range: ${Math.max(0,v-4.99).toFixed(1)}% – ${Math.min(100,v+4.99).toFixed(1)}%`;

      setTimeout(() => {
        document.getElementById('barFill').style.width = v + '%';
      }, 100);

      card.classList.add('show');
    } catch(e) {
      alert('Prediction failed: ' + e.message);
    }

    btn.classList.remove('loading');
    btn.textContent = '⚗ Predict Removal Efficiency';
  }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    d      = request.json
    metal  = d['metal']
    code   = 0.0 if metal == 'Cu' else 1.0
    inp    = scaler_X.transform([[d['time'], d['conc'], d['ph'], d['dosage'], d['temp'], code]])
    out    = net.predict(inp)
    removal = float(np.clip(scaler_Y.inverse_transform(out.reshape(-1,1))[0][0], 0, 100))
    return jsonify({'removal': round(removal, 2)})

if __name__ == '__main__':
    print("\n" + "="*45)
    print("  Metal Removal Predictor - Web Interface")
    print("  Open: http://localhost:5000")
    print("="*45 + "\n")
    app.run(debug=False, port=5000)
