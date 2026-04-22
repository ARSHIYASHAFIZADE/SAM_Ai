'use client'
import { useState } from 'react'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { RequireAuth } from '@/components/AuthProvider'
import { ResultDisplay, ExportReport as PrintReport, DS, type FormField } from '../heart/page'
import { SyringeIcon, AlertTriangleIcon } from '@/components/MedicalIcons'
import { saveAssessment } from '@/lib/assessmentHistory'
import { validateForm } from '@/lib/fieldValidation'
import { getTooltip } from '@/lib/fieldTooltips'
import FieldTooltip from '@/components/FieldTooltip'
import RecommendationsPanel from '@/components/RecommendationsPanel'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

const FIELDS = [
  { key: 'Age',               label: 'Age',               unit: 'yrs', placeholder: 'Years',         min: 1,  max: 120 },
  { key: 'Gender',            label: 'Gender',             options: ['Male','Female'] },
  { key: 'Polyuria',          label: 'Polyuria',           options: ['Yes','No'] },
  { key: 'Polydipsia',        label: 'Polydipsia',         options: ['Yes','No'] },
  { key: 'sudden_weight_loss',label: 'Sudden Weight Loss', options: ['Yes','No'] },
  { key: 'weakness',          label: 'Weakness',           options: ['Yes','No'] },
  { key: 'Polyphagia',        label: 'Polyphagia',         options: ['Yes','No'] },
  { key: 'Genital_thrush',    label: 'Genital Thrush',     options: ['Yes','No'] },
  { key: 'visual_blurring',   label: 'Visual Blurring',    options: ['Yes','No'] },
  { key: 'Itching',           label: 'Itching',            options: ['Yes','No'] },
  { key: 'Irritability',      label: 'Irritability',       options: ['Yes','No'] },
  { key: 'delayed_healing',   label: 'Delayed Healing',    options: ['Yes','No'] },
  { key: 'partial_paresis',   label: 'Partial Paresis',    options: ['Yes','No'] },
  { key: 'muscle_stiffness',  label: 'Muscle Stiffness',   options: ['Yes','No'] },
  { key: 'Alopecia',          label: 'Alopecia',           options: ['Yes','No'] },
  { key: 'Obesity',           label: 'Obesity',            options: ['Yes','No'] },
]

const YES_NO_FIELDS = new Set(['Polyuria','Polydipsia','sudden_weight_loss','weakness','Polyphagia','Genital_thrush','visual_blurring','Itching','Irritability','delayed_healing','partial_paresis','muscle_stiffness','Alopecia','Obesity'])

interface Result { prediction: number; probability: number; risk_level: string; model_version: string }

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
      const res = await fetch(`${API}/predict_male`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
        body: JSON.stringify(form),
      })
      if (!res.ok) throw new Error()
      const data = await res.json()
      setResult(data)
      saveAssessment({ type: 'diabetes-male', title: 'Diabetes (Male)', prediction: data.prediction, probability: data.probability, risk_level: data.risk_level })
    } catch { setError('Analysis failed. Please check inputs and try again.') }
    finally { setLoading(false) }
  }

  return (
    <div style={DS.page}>
      <Navbar />
      <div style={DS.wrap}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 8 }}>
            <div style={{ width: 50, height: 50, borderRadius: 15, background: 'rgba(96,165,250,0.12)', border: '1.5px solid rgba(96,165,250,0.30)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px rgba(96,165,250,0.12)' }}>
              <SyringeIcon size={24} color="#60a5fa" strokeWidth={1.7} />
            </div>
            <div>
              <p style={{ fontSize: 11, color: '#14b8a6', fontWeight: 700, letterSpacing: 1.5, textTransform: 'uppercase' }}>Endocrinology</p>
              <h1 style={{ fontSize: 26, fontWeight: 800, color: '#f1f5f9', letterSpacing: '-0.5px' }}>Diabetes Risk Assessment (Male)</h1>
            </div>
          </div>
          <p style={{ color: '#64748b', fontSize: 15, marginBottom: 36, marginLeft: 64 }}>Symptom driven screening across 16 clinical and lifestyle indicators.</p>
        </motion.div>

        {!result ? (
          <motion.form onSubmit={submit} noValidate initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <div style={DS.card}>
              <div style={{ ...DS.cardHead, background: 'linear-gradient(135deg, rgba(96,165,250,0.06), transparent)' }}>
                <h2 style={{ fontWeight: 700, fontSize: 15, color: '#f1f5f9' }}>Clinical Symptom Screening</h2>
                <p style={{ fontSize: 13, color: '#64748b', marginTop: 3 }}>Select your symptoms. Click <span style={{ color: '#14b8a6', fontWeight: 700 }}>ⓘ</span> for clinical descriptions.</p>
              </div>
              <div style={{ padding: 28, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 20 }}>
                {FIELDS.map(f => (
                  <div key={f.key}>
                    <label style={DS.label}>
                      {f.label}
                      <FieldTooltip data={getTooltip('diabetes-male', f.key)} />
                    </label>
                    {f.key === 'Age' ? (
                      <div style={{ position: 'relative' }}>
                        <input type="number" step="1" value={form[f.key]} onChange={e => setField(f.key, e.target.value)} placeholder={f.placeholder}
                          style={{ ...DS.input, paddingRight: 44, ...(errors[f.key] ? { borderColor: 'rgba(248,113,113,0.60)', boxShadow: '0 0 0 3px rgba(248,113,113,0.08)' } : {}) }} />
                        <span style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', fontSize: 10, color: '#475569', fontWeight: 700 }}>yrs</span>
                      </div>
                    ) : f.key === 'Gender' ? (
                      <select value={form[f.key]} onChange={e => setField(f.key, e.target.value)}
                        style={{ ...DS.select, ...(errors[f.key] ? { borderColor: 'rgba(248,113,113,0.60)', boxShadow: '0 0 0 3px rgba(248,113,113,0.08)' } : {}) }}>
                        <option value="">Select…</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                      </select>
                    ) : YES_NO_FIELDS.has(f.key) ? (
                      <select value={form[f.key]} onChange={e => setField(f.key, e.target.value)}
                        style={{ ...DS.select, ...(errors[f.key] ? { borderColor: 'rgba(248,113,113,0.60)', boxShadow: '0 0 0 3px rgba(248,113,113,0.08)' } : {}) }}>
                        <option value="">Select…</option>
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                      </select>
                    ) : null}
                    {errors[f.key] && <p style={{ fontSize: 11, color: '#f87171', marginTop: 5, animation: 'fade-in 0.2s ease' }}>⚠ {errors[f.key]}</p>}
                  </div>
                ))}
              </div>
              {error && <div style={{ margin: '0 28px 20px', padding: '12px 16px', background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.25)', borderRadius: 10, color: '#f87171', fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}><AlertTriangleIcon size={15} color="#f87171" strokeWidth={2} />{error}</div>}
              <div style={{ padding: '20px 28px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                <motion.button type="submit" disabled={loading}
                  style={{ padding: '13px 32px', borderRadius: 12, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 20px rgba(20,184,166,0.30)', display: 'flex', alignItems: 'center', gap: 8 }}
                  whileHover={!loading ? { y: -2 } : {}} whileTap={{ scale: 0.97 }}>
                  <SyringeIcon size={16} color="white" strokeWidth={2} />
                  {loading ? 'Analysing…' : 'Run Analysis →'}
                </motion.button>
              </div>
            </div>
          </motion.form>
        ) : (
          <div>
            <ResultDisplay result={result} onReset={() => setResult(null)} label={result.prediction === 1 ? 'Diabetes Risk Detected' : 'Low Diabetes Risk'} positiveColor="#60a5fa" negativeColor="#34d399" assessmentType="Diabetes (Male)" />
            <RecommendationsPanel type="diabetes-male" risk_level={result.risk_level} />
            <PrintReport result={result} assessmentType="Diabetes (Male)" formFields={FIELDS.map(f => ({ label: (f as {label:string}).label, value: form[(f as {key:string}).key] ?? '', unit: (f as {unit?:string}).unit }))} />
          </div>
        )}
      </div>
    </div>
  )
}

export default function DiabetesMalePage() { return <RequireAuth><Page /></RequireAuth> }
