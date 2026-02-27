'use client'
import { useState } from 'react'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { RequireAuth } from '@/components/AuthProvider'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

const FIELDS = [
  { key: 'mean_radius', label: 'Mean Radius', unit: 'μm' },
  { key: 'mean_texture', label: 'Mean Texture', unit: '' },
  { key: 'mean_perimeter', label: 'Mean Perimeter', unit: 'μm' },
  { key: 'mean_area', label: 'Mean Area', unit: 'μm²' },
  { key: 'mean_smoothness', label: 'Mean Smoothness', unit: '' },
  { key: 'mean_compactness', label: 'Mean Compactness', unit: '' },
  { key: 'mean_concavity', label: 'Mean Concavity', unit: '' },
  { key: 'mean_concave_points', label: 'Mean Concave Pts', unit: '' },
  { key: 'mean_symmetry', label: 'Mean Symmetry', unit: '' },
  { key: 'mean_fractal_dimension', label: 'Mean Fractal Dim', unit: '' },
  { key: 'radius_error', label: 'Radius Error', unit: '' },
  { key: 'texture_error', label: 'Texture Error', unit: '' },
  { key: 'perimeter_error', label: 'Perimeter Error', unit: '' },
  { key: 'area_error', label: 'Area Error', unit: '' },
  { key: 'smoothness_error', label: 'Smoothness Error', unit: '' },
  { key: 'compactness_error', label: 'Compactness Error', unit: '' },
  { key: 'concavity_error', label: 'Concavity Error', unit: '' },
  { key: 'concave_points_error', label: 'Concave Pts Error', unit: '' },
  { key: 'symmetry_error', label: 'Symmetry Error', unit: '' },
  { key: 'fractal_dimension_error', label: 'Fractal Dim Error', unit: '' },
  { key: 'worst_radius', label: 'Worst Radius', unit: 'μm' },
  { key: 'worst_texture', label: 'Worst Texture', unit: '' },
  { key: 'worst_perimeter', label: 'Worst Perimeter', unit: 'μm' },
  { key: 'worst_area', label: 'Worst Area', unit: 'μm²' },
  { key: 'worst_smoothness', label: 'Worst Smoothness', unit: '' },
  { key: 'worst_compactness', label: 'Worst Compactness', unit: '' },
  { key: 'worst_concavity', label: 'Worst Concavity', unit: '' },
  { key: 'worst_concave_points', label: 'Worst Concave Pts', unit: '' },
  { key: 'worst_symmetry', label: 'Worst Symmetry', unit: '' },
  { key: 'worst_fractal_dimension', label: 'Worst Fractal Dim', unit: '' },
]

interface Result { prediction: number; probability: number; risk_level: string; model_version: string; probability_breast_cancer?: number; radar_chart?: string; bar_chart?: string }

function Page() {
  const [form, setForm] = useState<Record<string, string>>(Object.fromEntries(FIELDS.map(f => [f.key, ''])))
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(''); setLoading(true)
    try {
      const res = await fetch(`${API}/detect_breast_cancer`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
        body: JSON.stringify(Object.fromEntries(Object.entries(form).map(([k, v]) => [k, Number(v)]))),
      })
      if (!res.ok) throw new Error()
      setResult(await res.json())
    } catch { setError('Analysis failed. Please check inputs.') }
    finally { setLoading(false) }
  }

  const riskBg = result?.risk_level === 'Low' ? '#f0fdf4' : result?.risk_level === 'Moderate' ? '#fffbeb' : '#fef2f2'
  const riskColor = result?.risk_level === 'Low' ? '#15803d' : result?.risk_level === 'Moderate' ? '#b45309' : '#dc2626'
  const isPositive = result?.prediction === 1

  return (
    <div style={{ minHeight: '100vh', background: '#fafafa' }}>
      <Navbar />
      <div style={{ maxWidth: 1000, margin: '0 auto', padding: '48px 24px 80px' }}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 32 }}>
          <div style={{ width: 48, height: 48, borderRadius: 14, background: '#f0fdf4', border: '1px solid #86efac', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>🔬</div>
          <div>
            <p style={{ fontSize: 12, color: '#0F9D9A', fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase' }}>Oncology</p>
            <h1 style={{ fontSize: 26, fontWeight: 800, color: '#09090b', letterSpacing: '-0.5px' }}>Breast Cancer Screening</h1>
            <p style={{ fontSize: 14, color: '#71717a', marginTop: 4 }}>30 cytology biomarkers from FNA cell nuclei — results emailed as PDF report</p>
          </div>
        </motion.div>

        {!result ? (
          <motion.form onSubmit={submit} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <div style={{ background: 'white', borderRadius: 20, border: '1px solid #f0f0f0', boxShadow: '0 2px 16px rgba(0,0,0,0.04)', overflow: 'hidden' }}>
              <div style={{ padding: '20px 28px', borderBottom: '1px solid #f4f4f5', background: 'linear-gradient(135deg, #f0fdf4, #fff)' }}>
                <h2 style={{ fontWeight: 700, fontSize: 16, color: '#09090b' }}>Cell Nucleus Measurements</h2>
                <p style={{ fontSize: 13, color: '#71717a', marginTop: 4 }}>Wisconsin Breast Cancer Dataset — 30 cytological features</p>
              </div>
              <div style={{ padding: 28, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 16 }}>
                {FIELDS.map(f => (
                  <div key={f.key}>
                    <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#52525b', marginBottom: 5, textTransform: 'uppercase', letterSpacing: 0.5 }}>{f.label}</label>
                    <div style={{ position: 'relative' }}>
                      <input type="number" step="any" required value={form[f.key]} onChange={e => setForm({ ...form, [f.key]: e.target.value })} placeholder="0.000"
                        style={{ width: '100%', padding: '9px 12px', paddingRight: f.unit ? 50 : 12, borderRadius: 9, border: '1.5px solid #e4e4e7', background: '#fafafa', fontSize: 13, color: '#09090b', transition: 'all 0.15s' }}
                        onFocus={e => { e.target.style.borderColor = '#0F9D9A'; e.target.style.background = 'white'; e.target.style.boxShadow = '0 0 0 3px rgba(15,157,154,0.12)' }}
                        onBlur={e => { e.target.style.borderColor = '#e4e4e7'; e.target.style.background = '#fafafa'; e.target.style.boxShadow = 'none' }} />
                      {f.unit && <span style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', fontSize: 9, color: '#a1a1aa', fontWeight: 700 }}>{f.unit}</span>}
                    </div>
                  </div>
                ))}
              </div>
              {error && <div style={{ margin: '0 28px 20px', padding: '12px 16px', background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 10, color: '#dc2626', fontSize: 14 }}>{error}</div>}
              <div style={{ padding: '20px 28px', borderTop: '1px solid #f4f4f5', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
                <p style={{ fontSize: 13, color: '#71717a' }}>📧 A detailed PDF report will be emailed after analysis.</p>
                <motion.button type="submit" disabled={loading} style={{ padding: '13px 32px', borderRadius: 12, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 16px rgba(15,157,154,0.3)' }}
                  whileHover={!loading ? { y: -2 } : {}} whileTap={{ scale: 0.97 }}>
                  {loading ? 'Analyzing…' : 'Run Screening →'}
                </motion.button>
              </div>
            </div>
          </motion.form>
        ) : (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div style={{ background: 'white', borderRadius: 24, border: '1px solid #f0f0f0', boxShadow: '0 8px 40px rgba(0,0,0,0.08)', overflow: 'hidden', marginBottom: 20 }}>
              <div style={{ padding: '28px 32px', borderBottom: '1px solid #f4f4f5', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
                <div>
                  <p style={{ fontSize: 12, color: '#71717a', fontWeight: 600, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 6 }}>Screening Result</p>
                  <h2 style={{ fontSize: 26, fontWeight: 800, color: isPositive ? '#dc2626' : '#16a34a', letterSpacing: '-0.5px' }}>{isPositive ? 'Malignant (Positive)' : 'Benign (Negative)'}</h2>
                </div>
                <div style={{ padding: '8px 18px', borderRadius: 100, background: riskBg, color: riskColor, fontWeight: 700, fontSize: 14, border: `1px solid ${riskColor}30`, boxShadow: `0 0 16px ${riskColor}20` }}>{result.risk_level} Risk</div>
              </div>
              <div style={{ padding: '28px 32px' }}>
                <div style={{ marginBottom: 24 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                    <span style={{ fontSize: 14, color: '#71717a', fontWeight: 600 }}>Confidence</span>
                    <span style={{ fontSize: 16, fontWeight: 800, color: '#09090b' }}>{result.probability.toFixed(1)}%</span>
                  </div>
                  <div style={{ height: 10, background: '#f4f4f5', borderRadius: 100, overflow: 'hidden' }}>
                    <motion.div style={{ height: '100%', borderRadius: 100, background: `linear-gradient(90deg, ${isPositive ? '#ef444480' : '#22c55e80'}, ${isPositive ? '#ef4444' : '#22c55e'})` }}
                      initial={{ width: 0 }} animate={{ width: `${result.probability}%` }} transition={{ duration: 1.2, ease: 'easeOut', delay: 0.2 }} />
                  </div>
                </div>
                {result.radar_chart && (
                  <div style={{ marginBottom: 20 }}>
                    <p style={{ fontSize: 13, fontWeight: 600, color: '#52525b', marginBottom: 10 }}>Radar Analysis</p>
                    <img src={`data:image/png;base64,${result.radar_chart}`} alt="Radar chart" style={{ width: '100%', borderRadius: 12, border: '1px solid #f4f4f5' }} />
                  </div>
                )}
                {result.bar_chart && (
                  <div style={{ marginBottom: 20 }}>
                    <p style={{ fontSize: 13, fontWeight: 600, color: '#52525b', marginBottom: 10 }}>Feature Comparison</p>
                    <img src={`data:image/png;base64,${result.bar_chart}`} alt="Bar chart" style={{ width: '100%', borderRadius: 12, border: '1px solid #f4f4f5' }} />
                  </div>
                )}
                <p style={{ fontSize: 13, color: '#71717a', marginBottom: 20 }}>Model v{result.model_version} · PDF report sent to your email</p>
                <motion.button onClick={() => setResult(null)} style={{ padding: '12px 28px', borderRadius: 12, background: 'white', border: '1.5px solid #e4e4e7', color: '#3f3f46', fontWeight: 600, fontSize: 14, cursor: 'pointer' }}
                  whileHover={{ y: -1, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} whileTap={{ scale: 0.98 }}>
                  ← New Screening
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default function BreastCancerPage() { return <RequireAuth><Page /></RequireAuth> }
