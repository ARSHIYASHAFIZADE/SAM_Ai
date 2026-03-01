'use client'
import { useState } from 'react'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { RequireAuth } from '@/components/AuthProvider'
import { HeartECGIcon, PrinterIcon, RefreshIcon, AlertTriangleIcon, CheckCircleIcon } from '@/components/MedicalIcons'
import { saveAssessment } from '@/lib/assessmentHistory'
import { getTooltip } from '@/lib/fieldTooltips'
import { validateForm } from '@/lib/fieldValidation'
import FieldTooltip from '@/components/FieldTooltip'
import RecommendationsPanel from '@/components/RecommendationsPanel'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

type Option = { label: string; value: string }
type Field = { key: string; label: string; placeholder?: string; unit?: string; options?: Option[]; min?: number; max?: number }

const FIELDS: Field[] = [
  { key: 'age',      label: 'Age',                     placeholder: 'Years',   unit: 'yrs',   min: 1,   max: 120 },
  { key: 'sex',      label: 'Sex',                     options: [{ label: 'Male', value: '1' }, { label: 'Female', value: '0' }] },
  { key: 'cp',       label: 'Chest Pain Type',         placeholder: '0–3',                    min: 0,   max: 3 },
  { key: 'trestbps', label: 'Resting BP',              placeholder: 'mm Hg',  unit: 'mmHg',  min: 60,  max: 250 },
  { key: 'chol',     label: 'Cholesterol',             placeholder: 'mg/dL',  unit: 'mg/dL', min: 100, max: 700 },
  { key: 'fbs',      label: 'Fasting Blood Sugar >120', options: [{ label: 'Yes (>120 mg/dL)', value: '1' }, { label: 'No', value: '0' }] },
  { key: 'restecg',  label: 'Resting ECG',             placeholder: '0–2',                    min: 0,   max: 2 },
  { key: 'thalach',  label: 'Max Heart Rate',          placeholder: 'bpm',    unit: 'bpm',   min: 40,  max: 250 },
  { key: 'exang',    label: 'Exercise Angina',         options: [{ label: 'Yes', value: '1' }, { label: 'No', value: '0' }] },
  { key: 'oldpeak',  label: 'ST Depression',           placeholder: '0.0–6.2',                min: 0,   max: 10 },
  { key: 'slope',    label: 'ST Slope',                placeholder: '0–2',                    min: 0,   max: 2 },
  { key: 'ca',       label: 'Major Vessels',           placeholder: '0–4',                    min: 0,   max: 4 },
  { key: 'thal',     label: 'Thalassemia',             placeholder: '1–3',                    min: 1,   max: 3 },
]

interface Result { prediction: number; probability: number; risk_level: string; model_version: string }

// ── Shared dark styles ─────────────────────────────────────
export const DS = {
  page:     { minHeight: '100vh', background: '#0a0f1e' },
  wrap:     { maxWidth: 900, margin: '0 auto', padding: '48px 24px 80px' },
  card:     { background: '#111827', borderRadius: 22, border: '1px solid rgba(255,255,255,0.08)', boxShadow: '0 4px 24px rgba(0,0,0,0.30)', overflow: 'hidden' },
  cardHead: { padding: '20px 28px', borderBottom: '1px solid rgba(255,255,255,0.06)' },
  label:    { display: 'block' as const, fontSize: 12, fontWeight: 700, color: '#64748b', marginBottom: 6, textTransform: 'uppercase' as const, letterSpacing: '0.5px' },
  input:    { width: '100%', padding: '10px 14px', borderRadius: 10, border: '1.5px solid rgba(255,255,255,0.10)', background: 'rgba(255,255,255,0.04)', fontSize: 14, color: '#f1f5f9', transition: 'all 0.15s' },
  select:   { width: '100%', padding: '10px 14px', borderRadius: 10, border: '1.5px solid rgba(255,255,255,0.10)', background: '#1a2236', fontSize: 14, color: '#f1f5f9', transition: 'all 0.15s', cursor: 'pointer', appearance: 'none' as const },
}

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
      const res = await fetch(`${API}/detect_heart`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ data: Object.fromEntries(Object.entries(form).map(([k, v]) => [k, Number(v)])) }),
      })
      if (!res.ok) throw new Error('Prediction failed')
      const data = await res.json()
      setResult(data)
      saveAssessment({ type: 'heart', title: 'Heart Disease', prediction: data.prediction, probability: data.probability, risk_level: data.risk_level })
    } catch { setError('Analysis failed. Please check your inputs and try again.') }
    finally { setLoading(false) }
  }

  return (
    <div style={DS.page}>
      <Navbar />
      <div style={DS.wrap}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 8 }}>
            <div style={{ width: 50, height: 50, borderRadius: 15, background: 'rgba(248,113,113,0.12)', border: '1.5px solid rgba(248,113,113,0.30)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px rgba(248,113,113,0.15)' }}>
              <HeartECGIcon size={24} color="#f87171" strokeWidth={1.7} />
            </div>
            <div>
              <p style={{ fontSize: 11, color: '#14b8a6', fontWeight: 700, letterSpacing: 1.5, textTransform: 'uppercase' }}>Cardiology</p>
              <h1 style={{ fontSize: 26, fontWeight: 800, color: '#f1f5f9', letterSpacing: '-0.5px' }}>Heart Disease Risk Assessment</h1>
            </div>
          </div>
          <p style={{ color: '#64748b', fontSize: 15, marginBottom: 36, marginLeft: 64 }}>Enter 13 clinical vitals to assess cardiovascular risk using AI.</p>
        </motion.div>

        {!result ? (
          <motion.form onSubmit={submit} noValidate initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <div style={DS.card}>
              <div style={{ ...DS.cardHead, background: 'linear-gradient(135deg, rgba(248,113,113,0.06), transparent)' }}>
                <h2 style={{ fontWeight: 700, fontSize: 15, color: '#f1f5f9' }}>Patient Clinical Data</h2>
                <p style={{ fontSize: 13, color: '#64748b', marginTop: 3 }}>Fill in all 13 cardiovascular markers. Click <span style={{ color: '#14b8a6', fontWeight: 700 }}>ⓘ</span> for clinical guidance.</p>
              </div>
              <div style={{ padding: 28, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 20 }}>
                {FIELDS.map(f => (
                  <div key={f.key}>
                    <label style={DS.label}>
                      {f.label}
                      <FieldTooltip data={getTooltip('heart', f.key)} />
                    </label>
                    <div style={{ position: 'relative' }}>
                      {f.options ? (
                        <select value={form[f.key]} onChange={e => setField(f.key, e.target.value)}
                          style={{ ...DS.select, ...(errors[f.key] ? { borderColor: 'rgba(248,113,113,0.60)', boxShadow: '0 0 0 3px rgba(248,113,113,0.08)' } : {}) }}>
                          <option value="">Select…</option>
                          {f.options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>
                      ) : (
                        <input type="number" step="any" value={form[f.key]}
                          onChange={e => setField(f.key, e.target.value)}
                          placeholder={f.placeholder}
                          style={{ ...DS.input, paddingRight: f.unit ? 52 : 14, ...(errors[f.key] ? { borderColor: 'rgba(248,113,113,0.60)', boxShadow: '0 0 0 3px rgba(248,113,113,0.08)' } : {}) }} />
                      )}
                      {f.unit && !f.options && <span style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', fontSize: 10, color: '#475569', fontWeight: 700, letterSpacing: 0.3 }}>{f.unit}</span>}
                    </div>
                    {errors[f.key] && <p style={{ fontSize: 11, color: '#f87171', marginTop: 5, animation: 'fade-in 0.2s ease' }}>⚠ {errors[f.key]}</p>}
                  </div>
                ))}
              </div>
              {error && (
                <div style={{ margin: '0 28px 20px', padding: '12px 16px', background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.25)', borderRadius: 10, color: '#f87171', fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <AlertTriangleIcon size={15} color="#f87171" strokeWidth={2} />{error}
                </div>
              )}
              <div style={{ padding: '20px 28px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                <motion.button type="submit" disabled={loading}
                  style={{ padding: '13px 32px', borderRadius: 12, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 20px rgba(20,184,166,0.30)', display: 'flex', alignItems: 'center', gap: 8 }}
                  whileHover={!loading ? { y: -2, boxShadow: '0 8px 28px rgba(20,184,166,0.40)' } : {}} whileTap={{ scale: 0.97 }}>
                  <HeartECGIcon size={16} color="white" strokeWidth={2} />
                  {loading ? 'Analysing…' : 'Run Analysis →'}
                </motion.button>
              </div>
            </div>
          </motion.form>
        ) : (
          <div>
            <ResultDisplay result={result} onReset={() => setResult(null)}
              label={result.prediction === 1 ? 'Disease Detected' : 'No Disease Detected'}
              positiveColor="#f87171" negativeColor="#34d399"
              assessmentType="Heart Disease" />
            <RecommendationsPanel type="heart" risk_level={result.risk_level} />
            <PrintReport result={result} assessmentType="Heart Disease" />
          </div>
        )}
      </div>
    </div>
  )
}

// ── Shared ResultDisplay (exported & used by all predict pages) ──
export function ResultDisplay({ result, onReset, label, positiveColor, negativeColor, assessmentType }: {
  result: Result; onReset: () => void; label: string;
  positiveColor: string; negativeColor: string; assessmentType?: string
}) {
  const isPositive = result.prediction === 1
  const color = isPositive ? positiveColor : negativeColor
  const riskColor = result.risk_level === 'Low' ? '#34d399' : result.risk_level === 'Moderate' ? '#fbbf24' : '#f87171'
  const riskBg    = result.risk_level === 'Low' ? 'rgba(52,211,153,0.10)' : result.risk_level === 'Moderate' ? 'rgba(251,191,36,0.10)' : 'rgba(248,113,113,0.10)'
  const riskBorder = result.risk_level === 'Low' ? 'rgba(52,211,153,0.25)' : result.risk_level === 'Moderate' ? 'rgba(251,191,36,0.25)' : 'rgba(248,113,113,0.25)'

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      style={{ background: '#111827', borderRadius: 22, border: '1px solid rgba(255,255,255,0.08)', boxShadow: '0 8px 40px rgba(0,0,0,0.30)', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '28px 32px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, background: `linear-gradient(135deg, rgba(${isPositive ? (color === '#f87171' ? '248,113,113' : '52,211,153') : '52,211,153'},0.06), transparent)` }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            {isPositive
              ? <AlertTriangleIcon size={18} color={positiveColor} strokeWidth={2} />
              : <CheckCircleIcon size={18} color={negativeColor} strokeWidth={2} />
            }
            <p style={{ fontSize: 11, color: '#64748b', fontWeight: 600, letterSpacing: 1, textTransform: 'uppercase' }}>Analysis Result</p>
          </div>
          <h2 style={{ fontSize: 28, fontWeight: 800, color, letterSpacing: '-0.5px' }}>{label}</h2>
        </div>
        <div style={{ padding: '8px 20px', borderRadius: 100, background: riskBg, color: riskColor, fontWeight: 700, fontSize: 14, border: `1px solid ${riskBorder}` }}>
          {result.risk_level} Risk
        </div>
      </div>

      <div style={{ padding: '32px' }}>
        {/* Confidence bar */}
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
            <span style={{ fontSize: 13, color: '#64748b', fontWeight: 600 }}>Model Confidence Score</span>
            <span style={{ fontSize: 16, fontWeight: 800, color: '#f1f5f9' }}>{result.probability.toFixed(1)}%</span>
          </div>
          <div style={{ height: 10, background: 'rgba(255,255,255,0.06)', borderRadius: 100, overflow: 'hidden' }}>
            <motion.div
              style={{ height: '100%', borderRadius: 100, background: `linear-gradient(90deg, ${color}60, ${color})` }}
              initial={{ width: 0 }} animate={{ width: `${result.probability}%` }}
              transition={{ duration: 1.2, ease: 'easeOut', delay: 0.2 }} />
          </div>
        </div>

        {/* Print-only report div */}
        <div className="print-report" style={{ display: 'none' }}>
          <div style={{ marginBottom: 24, paddingBottom: 16, borderBottom: '2px solid #0F9D9A' }}>
            <h1 style={{ fontSize: 22, fontWeight: 800 }}>SAM AI Clinical Assessment Report</h1>
            {assessmentType && <p style={{ fontSize: 14, marginTop: 4 }}>Module: {assessmentType}</p>}
            <p style={{ fontSize: 13, color: '#666', marginTop: 2 }}>Generated: {new Date().toLocaleString()}</p>
          </div>
          <div style={{ marginBottom: 16 }}>
            <p style={{ fontSize: 18, fontWeight: 800, marginBottom: 6 }}>Result: {label}</p>
            <span className={`print-risk-${result.risk_level.toLowerCase()}`} style={{ padding: '4px 14px', borderRadius: 6, fontSize: 13, fontWeight: 700 }}>{result.risk_level} Risk</span>
          </div>
          <div style={{ marginBottom: 16 }}>
            <p style={{ fontSize: 14, marginBottom: 8, fontWeight: 600 }}>Confidence Score: {result.probability.toFixed(1)}%</p>
            <div style={{ height: 12, background: '#f0f0f0', borderRadius: 6, overflow: 'hidden', marginBottom: 4 }}>
              <div className="print-bar" style={{ height: '100%', width: `${result.probability}%`, borderRadius: 6 }} />
            </div>
          </div>
          {result.model_version && <p style={{ fontSize: 12, color: '#888' }}>Model version: {result.model_version}</p>}
          <p style={{ fontSize: 11, color: '#999', marginTop: 24, paddingTop: 12, borderTop: '1px solid #ddd' }}>
            SAM AI is for educational purposes only. This report does not constitute a medical diagnosis. Always consult a qualified healthcare professional.
          </p>
        </div>

        <p style={{ fontSize: 13, color: '#475569', marginBottom: 28 }}>
          {result.model_version && `Model v${result.model_version} · `}For educational purposes only
        </p>

        <motion.button onClick={onReset}
          style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '11px 24px', borderRadius: 11, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.10)', color: '#94a3b8', fontWeight: 600, fontSize: 14, cursor: 'pointer', transition: 'all 0.15s' }}
          whileHover={{ y: -1, background: 'rgba(255,255,255,0.08)' as string }} whileTap={{ scale: 0.98 }}>
          <RefreshIcon size={14} color="#94a3b8" strokeWidth={2.2} />
          New Assessment
        </motion.button>
      </div>
    </motion.div>
  )
}

// ── Print Report Button ────────────────────────────────────
export function PrintReport({ result: _result, assessmentType: _at }: { result: Result; assessmentType: string }) {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
      style={{ marginTop: 16, display: 'flex', justifyContent: 'flex-end' }}>
      <button onClick={() => window.print()}
        style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '10px 20px', borderRadius: 10, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.09)', color: '#64748b', fontWeight: 600, fontSize: 13, cursor: 'pointer', transition: 'all 0.15s' }}
        onMouseEnter={e => { e.currentTarget.style.color = '#94a3b8'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.18)' }}
        onMouseLeave={e => { e.currentTarget.style.color = '#64748b'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.09)' }}>
        <PrinterIcon size={14} color="currentColor" strokeWidth={2} />
        Print Report
      </button>
    </motion.div>
  )
}

export { DS as DARK_STYLES }
export default function HeartPage() { return <RequireAuth><Page /></RequireAuth> }
