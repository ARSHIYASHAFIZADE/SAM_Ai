'use client'
import { useState } from 'react'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { RequireAuth } from '@/components/AuthProvider'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

const FIELDS = [
  { key: 'age', label: 'Age', placeholder: 'Years', unit: 'yrs' },
  { key: 'sex', label: 'Sex', placeholder: '1 = Male, 0 = Female', unit: '' },
  { key: 'cp', label: 'Chest Pain Type', placeholder: '0–3', unit: '' },
  { key: 'trestbps', label: 'Resting BP', placeholder: 'mm Hg', unit: 'mmHg' },
  { key: 'chol', label: 'Cholesterol', placeholder: 'mg/dL', unit: 'mg/dL' },
  { key: 'fbs', label: 'Fasting Blood Sugar >120', placeholder: '1 = Yes, 0 = No', unit: '' },
  { key: 'restecg', label: 'Resting ECG', placeholder: '0–2', unit: '' },
  { key: 'thalach', label: 'Max Heart Rate', placeholder: 'bpm', unit: 'bpm' },
  { key: 'exang', label: 'Exercise Angina', placeholder: '1 = Yes, 0 = No', unit: '' },
  { key: 'oldpeak', label: 'ST Depression', placeholder: '0.0–6.2', unit: '' },
  { key: 'slope', label: 'ST Slope', placeholder: '0–2', unit: '' },
  { key: 'ca', label: 'Major Vessels', placeholder: '0–4', unit: '' },
  { key: 'thal', label: 'Thalassemia', placeholder: '1–3', unit: '' },
]

interface Result { prediction: number; probability: number; risk_level: string; model_version: string }

function Page() {
  const [form, setForm] = useState<Record<string, string>>(Object.fromEntries(FIELDS.map(f => [f.key, ''])))
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(''); setLoading(true)
    try {
      const res = await fetch(`${API}/detect_heart`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ data: Object.fromEntries(Object.entries(form).map(([k, v]) => [k, Number(v)])) }),
      })
      if (!res.ok) throw new Error('Prediction failed')
      setResult(await res.json())
    } catch { setError('Analysis failed. Please check your inputs and try again.') }
    finally { setLoading(false) }
  }

  return (
    <div style={{ minHeight: '100vh', background: '#fafafa' }}>
      <Navbar />
      <div style={{ maxWidth: 900, margin: '0 auto', padding: '48px 24px 80px' }}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 8 }}>
            <div style={{ width: 48, height: 48, borderRadius: 14, background: '#fef2f2', border: '1px solid #fca5a5', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>🫀</div>
            <div>
              <p style={{ fontSize: 12, color: '#0F9D9A', fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase' }}>Cardiology</p>
              <h1 style={{ fontSize: 26, fontWeight: 800, color: '#09090b', letterSpacing: '-0.5px' }}>Heart Disease Risk Assessment</h1>
            </div>
          </div>
          <p style={{ color: '#71717a', fontSize: 15, marginBottom: 36, marginLeft: 62 }}>Enter 13 clinical vitals to assess cardiovascular risk using AI.</p>
        </motion.div>

        {!result ? (
          <motion.form onSubmit={submit} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <div style={{ background: 'white', borderRadius: 20, border: '1px solid #f0f0f0', boxShadow: '0 2px 16px rgba(0,0,0,0.04)', overflow: 'hidden' }}>
              <div style={{ padding: '20px 28px', borderBottom: '1px solid #f4f4f5', background: 'linear-gradient(135deg, #fef2f2, #fff)' }}>
                <h2 style={{ fontWeight: 700, fontSize: 16, color: '#09090b' }}>Patient Clinical Data</h2>
                <p style={{ fontSize: 13, color: '#71717a', marginTop: 4 }}>Fill in all 13 cardiovascular markers</p>
              </div>
              <div style={{ padding: 28, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 20 }}>
                {FIELDS.map(f => (
                  <div key={f.key}>
                    <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#52525b', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>{f.label}</label>
                    <div style={{ position: 'relative' }}>
                      <input
                        type="number" step="any" required value={form[f.key]}
                        onChange={e => setForm({ ...form, [f.key]: e.target.value })}
                        placeholder={f.placeholder}
                        style={{ width: '100%', padding: '10px 14px', borderRadius: 10, border: '1.5px solid #e4e4e7', background: '#fafafa', fontSize: 14, color: '#09090b', transition: 'all 0.15s', paddingRight: f.unit ? 52 : 14 }}
                        onFocus={e => { e.target.style.borderColor = '#0F9D9A'; e.target.style.background = 'white'; e.target.style.boxShadow = '0 0 0 3px rgba(15,157,154,0.12)' }}
                        onBlur={e => { e.target.style.borderColor = '#e4e4e7'; e.target.style.background = '#fafafa'; e.target.style.boxShadow = 'none' }}
                      />
                      {f.unit && <span style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', fontSize: 11, color: '#a1a1aa', fontWeight: 600 }}>{f.unit}</span>}
                    </div>
                  </div>
                ))}
              </div>
              {error && <div style={{ margin: '0 28px 20px', padding: '12px 16px', background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 10, color: '#dc2626', fontSize: 14 }}>{error}</div>}
              <div style={{ padding: '20px 28px', borderTop: '1px solid #f4f4f5' }}>
                <motion.button type="submit" disabled={loading}
                  style={{ padding: '13px 32px', borderRadius: 12, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 16px rgba(15,157,154,0.3)' }}
                  whileHover={!loading ? { y: -2, boxShadow: '0 8px 24px rgba(15,157,154,0.4)' } : {}} whileTap={{ scale: 0.97 }}>
                  {loading ? 'Analyzing…' : 'Run Analysis →'}
                </motion.button>
              </div>
            </div>
          </motion.form>
        ) : (
          <ResultDisplay result={result} onReset={() => setResult(null)} label={result.prediction === 1 ? 'Disease Detected' : 'No Disease Detected'} positiveColor="#dc2626" negativeColor="#16a34a" />
        )}
      </div>
    </div>
  )
}

function ResultDisplay({ result, onReset, label, positiveColor, negativeColor }: { result: Result; onReset: () => void; label: string; positiveColor: string; negativeColor: string }) {
  const isPositive = result.prediction === 1
  const color = isPositive ? positiveColor : negativeColor
  const riskBg = result.risk_level === 'Low' ? '#f0fdf4' : result.risk_level === 'Moderate' ? '#fffbeb' : '#fef2f2'
  const riskColor = result.risk_level === 'Low' ? '#15803d' : result.risk_level === 'Moderate' ? '#b45309' : '#dc2626'

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ background: 'white', borderRadius: 24, border: '1px solid #f0f0f0', boxShadow: '0 8px 40px rgba(0,0,0,0.08)', overflow: 'hidden' }}>
      <div style={{ padding: '32px 36px', borderBottom: '1px solid #f4f4f5', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
        <div>
          <p style={{ fontSize: 12, color: '#71717a', fontWeight: 600, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 6 }}>Analysis Result</p>
          <h2 style={{ fontSize: 28, fontWeight: 800, color, letterSpacing: '-0.5px' }}>{label}</h2>
        </div>
        <div style={{ padding: '8px 18px', borderRadius: 100, background: riskBg, color: riskColor, fontWeight: 700, fontSize: 14, border: `1px solid ${riskColor}30`, boxShadow: `0 0 16px ${riskColor}20` }}>
          {result.risk_level} Risk
        </div>
      </div>
      <div style={{ padding: '36px' }}>
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
            <span style={{ fontSize: 14, color: '#71717a', fontWeight: 600 }}>Confidence Score</span>
            <span style={{ fontSize: 16, fontWeight: 800, color: '#09090b' }}>{result.probability.toFixed(1)}%</span>
          </div>
          <div style={{ height: 10, background: '#f4f4f5', borderRadius: 100, overflow: 'hidden' }}>
            <motion.div
              style={{ height: '100%', borderRadius: 100, background: `linear-gradient(90deg, ${color}80, ${color})` }}
              initial={{ width: 0 }} animate={{ width: `${result.probability}%` }}
              transition={{ duration: 1.2, ease: 'easeOut', delay: 0.2 }}
            />
          </div>
        </div>
        <p style={{ fontSize: 13, color: '#71717a', marginBottom: 28 }}>Model v{result.model_version} · For educational purposes only</p>
        <motion.button onClick={onReset}
          style={{ padding: '12px 28px', borderRadius: 12, background: 'white', border: '1.5px solid #e4e4e7', color: '#3f3f46', fontWeight: 600, fontSize: 14, cursor: 'pointer', transition: 'all 0.15s' }}
          whileHover={{ y: -1, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} whileTap={{ scale: 0.98 }}>
          ← New Assessment
        </motion.button>
      </div>
    </motion.div>
  )
}

export { ResultDisplay }
export default function HeartPage() { return <RequireAuth><Page /></RequireAuth> }
