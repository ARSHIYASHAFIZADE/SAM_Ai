'use client'
import { useState } from 'react'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { RequireAuth } from '@/components/AuthProvider'
import { ResultDisplay, PrintReport, DS } from '../heart/page'
import { MicroscopeIcon, AlertTriangleIcon } from '@/components/MedicalIcons'
import { saveAssessment } from '@/lib/assessmentHistory'
import { validateForm } from '@/lib/fieldValidation'
import { getTooltip } from '@/lib/fieldTooltips'
import FieldTooltip from '@/components/FieldTooltip'
import RecommendationsPanel from '@/components/RecommendationsPanel'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

const FIELDS = [
  { key: 'mean_radius',            label: 'Mean Radius',          unit: 'μm',  min: 0, max: 50 },
  { key: 'mean_texture',           label: 'Mean Texture',         unit: '',    min: 0, max: 80 },
  { key: 'mean_perimeter',         label: 'Mean Perimeter',       unit: 'μm',  min: 0, max: 500 },
  { key: 'mean_area',              label: 'Mean Area',            unit: 'μm²', min: 0, max: 5000 },
  { key: 'mean_smoothness',        label: 'Mean Smoothness',      unit: '',    min: 0, max: 1 },
  { key: 'mean_compactness',       label: 'Mean Compactness',     unit: '',    min: 0, max: 2 },
  { key: 'mean_concavity',         label: 'Mean Concavity',       unit: '',    min: 0, max: 2 },
  { key: 'mean_concave_points',    label: 'Mean Concave Pts',     unit: '',    min: 0, max: 1 },
  { key: 'mean_symmetry',          label: 'Mean Symmetry',        unit: '',    min: 0, max: 1 },
  { key: 'mean_fractal_dimension', label: 'Mean Fractal Dim',     unit: '',    min: 0, max: 1 },
  { key: 'radius_error',           label: 'Radius Error',         unit: '',    min: 0, max: 10 },
  { key: 'texture_error',          label: 'Texture Error',        unit: '',    min: 0, max: 20 },
  { key: 'perimeter_error',        label: 'Perimeter Error',      unit: '',    min: 0, max: 100 },
  { key: 'area_error',             label: 'Area Error',           unit: '',    min: 0, max: 1000 },
  { key: 'smoothness_error',       label: 'Smoothness Error',     unit: '',    min: 0, max: 0.1 },
  { key: 'compactness_error',      label: 'Compactness Error',    unit: '',    min: 0, max: 0.5 },
  { key: 'concavity_error',        label: 'Concavity Error',      unit: '',    min: 0, max: 0.5 },
  { key: 'concave_points_error',   label: 'Concave Pts Error',    unit: '',    min: 0, max: 0.2 },
  { key: 'symmetry_error',         label: 'Symmetry Error',       unit: '',    min: 0, max: 0.1 },
  { key: 'fractal_dimension_error',label: 'Fractal Dim Error',    unit: '',    min: 0, max: 0.05 },
  { key: 'worst_radius',           label: 'Worst Radius',         unit: 'μm',  min: 0, max: 50 },
  { key: 'worst_texture',          label: 'Worst Texture',        unit: '',    min: 0, max: 100 },
  { key: 'worst_perimeter',        label: 'Worst Perimeter',      unit: 'μm',  min: 0, max: 500 },
  { key: 'worst_area',             label: 'Worst Area',           unit: 'μm²', min: 0, max: 5000 },
  { key: 'worst_smoothness',       label: 'Worst Smoothness',     unit: '',    min: 0, max: 1 },
  { key: 'worst_compactness',      label: 'Worst Compactness',    unit: '',    min: 0, max: 2 },
  { key: 'worst_concavity',        label: 'Worst Concavity',      unit: '',    min: 0, max: 2 },
  { key: 'worst_concave_points',   label: 'Worst Concave Pts',    unit: '',    min: 0, max: 1 },
  { key: 'worst_symmetry',         label: 'Worst Symmetry',       unit: '',    min: 0, max: 1 },
  { key: 'worst_fractal_dimension',label: 'Worst Fractal Dim',    unit: '',    min: 0, max: 1 },
]

interface Result { prediction: number; probability: number; risk_level: string; model_version: string; probability_breast_cancer?: number; radar_chart?: string; bar_chart?: string }

function Page() {
  const [form, setForm] = useState<Record<string, string>>(Object.fromEntries(FIELDS.map(f => [f.key, ''])))
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [errors, setErrors] = useState<Record<string, string>>({})

  const setField = (key: string, val: string) => {
    setForm(f => ({ ...f, [key]: val }))
    if (errors[key]) setErrors(e => { const n = { ...e }; delete n[key]; return n })
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('')
    const errs = validateForm(form, FIELDS)
    if (Object.keys(errs).length > 0) { setErrors(errs); return }
    setErrors({}); setLoading(true)
    try {
      const res = await fetch(`${API}/detect_breast_cancer`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
        body: JSON.stringify(Object.fromEntries(Object.entries(form).map(([k, v]) => [k, Number(v)]))),
      })
      if (!res.ok) throw new Error()
      const data = await res.json()
      setResult(data)
      saveAssessment({ type: 'breast-cancer', title: 'Breast Cancer', prediction: data.prediction, probability: data.probability, risk_level: data.risk_level })
    } catch { setError('Analysis failed. Please check inputs.') }
    finally { setLoading(false) }
  }

  return (
    <div style={DS.page}>
      <Navbar />
      <div style={DS.wrap}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 8 }}>
            <div style={{ width: 50, height: 50, borderRadius: 15, background: 'rgba(52,211,153,0.12)', border: '1.5px solid rgba(52,211,153,0.30)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px rgba(52,211,153,0.12)' }}>
              <MicroscopeIcon size={24} color="#34d399" strokeWidth={1.7} />
            </div>
            <div>
              <p style={{ fontSize: 11, color: '#14b8a6', fontWeight: 700, letterSpacing: 1.5, textTransform: 'uppercase' }}>Oncology</p>
              <h1 style={{ fontSize: 26, fontWeight: 800, color: '#f1f5f9', letterSpacing: '-0.5px' }}>Breast Cancer Classification</h1>
            </div>
          </div>
          <p style={{ color: '#64748b', fontSize: 15, marginBottom: 36, marginLeft: 64 }}>30 FNA cytology biomarkers — malignant vs benign nucleus classification.</p>
        </motion.div>

        {!result ? (
          <motion.form onSubmit={submit} noValidate initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <div style={DS.card}>
              <div style={{ ...DS.cardHead, background: 'linear-gradient(135deg, rgba(52,211,153,0.06), transparent)' }}>
                <h2 style={{ fontWeight: 700, fontSize: 15, color: '#f1f5f9' }}>FNA Cytology Measurements</h2>
                <p style={{ fontSize: 13, color: '#64748b', marginTop: 3 }}>Enter 30 nucleus measurements from FNA analysis. Click <span style={{ color: '#14b8a6', fontWeight: 700 }}>ⓘ</span> on mean features for definitions.</p>
              </div>
              <div style={{ padding: 28, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 16 }}>
                {FIELDS.map(f => (
                  <div key={f.key}>
                    <label style={DS.label}>
                      {f.label}
                      <FieldTooltip data={getTooltip('breast-cancer', f.key)} />
                    </label>
                    <div style={{ position: 'relative' }}>
                      <input type="number" step="any" value={form[f.key]}
                        onChange={e => setField(f.key, e.target.value)}
                        style={{ ...DS.input, paddingRight: f.unit ? 44 : 14, ...(errors[f.key] ? { borderColor: 'rgba(248,113,113,0.60)', boxShadow: '0 0 0 3px rgba(248,113,113,0.08)' } : {}) }} />
                      {f.unit && <span style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', fontSize: 9, color: '#475569', fontWeight: 700 }}>{f.unit}</span>}
                    </div>
                    {errors[f.key] && <p style={{ fontSize: 11, color: '#f87171', marginTop: 5, animation: 'fade-in 0.2s ease' }}>⚠ {errors[f.key]}</p>}
                  </div>
                ))}
              </div>
              {error && <div style={{ margin: '0 28px 20px', padding: '12px 16px', background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.25)', borderRadius: 10, color: '#f87171', fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}><AlertTriangleIcon size={15} color="#f87171" strokeWidth={2} />{error}</div>}
              <div style={{ padding: '20px 28px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                <motion.button type="submit" disabled={loading}
                  style={{ padding: '13px 32px', borderRadius: 12, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 20px rgba(20,184,166,0.30)', display: 'flex', alignItems: 'center', gap: 8 }}
                  whileHover={!loading ? { y: -2 } : {}} whileTap={{ scale: 0.97 }}>
                  <MicroscopeIcon size={16} color="white" strokeWidth={2} />
                  {loading ? 'Classifying…' : 'Run Classification →'}
                </motion.button>
              </div>
            </div>
          </motion.form>
        ) : (
          <div>
            <ResultDisplay result={result} onReset={() => setResult(null)} label={result.prediction === 1 ? 'Malignant Classification' : 'Benign Classification'} positiveColor="#f87171" negativeColor="#34d399" assessmentType="Breast Cancer" />
            {result.radar_chart && (
              <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
                style={{ marginTop: 20, background: '#111827', borderRadius: 18, border: '1px solid rgba(255,255,255,0.07)', padding: 24 }}>
                <p style={{ fontSize: 13, fontWeight: 700, color: '#94a3b8', marginBottom: 12 }}>Radar Analysis</p>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`data:image/png;base64,${result.radar_chart}`} alt="Radar chart" style={{ maxWidth: '100%', borderRadius: 12 }} />
              </motion.div>
            )}
            <RecommendationsPanel type="breast-cancer" risk_level={result.risk_level} />
            <PrintReport result={result} assessmentType="Breast Cancer" />
          </div>
        )}
      </div>
    </div>
  )
}

export default function BreastCancerPage() { return <RequireAuth><Page /></RequireAuth> }
