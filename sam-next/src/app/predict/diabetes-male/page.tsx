'use client'
import { useState } from 'react'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { RequireAuth } from '@/components/AuthProvider'
import { ResultDisplay } from '../heart/page'
import { SyringeIcon } from '@/components/MedicalIcons'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

const FIELDS = [
  { key: 'Age', label: 'Age', unit: 'yrs', placeholder: 'Years' },
  { key: 'Gender', label: 'Gender', unit: '', placeholder: 'Male/Female' },
  { key: 'Polyuria', label: 'Polyuria', unit: '', placeholder: 'Yes/No' },
  { key: 'Polydipsia', label: 'Polydipsia', unit: '', placeholder: 'Yes/No' },
  { key: 'sudden_weight_loss', label: 'Sudden Weight Loss', unit: '', placeholder: 'Yes/No' },
  { key: 'weakness', label: 'Weakness', unit: '', placeholder: 'Yes/No' },
  { key: 'Polyphagia', label: 'Polyphagia', unit: '', placeholder: 'Yes/No' },
  { key: 'Genital_thrush', label: 'Genital Thrush', unit: '', placeholder: 'Yes/No' },
  { key: 'visual_blurring', label: 'Visual Blurring', unit: '', placeholder: 'Yes/No' },
  { key: 'Itching', label: 'Itching', unit: '', placeholder: 'Yes/No' },
  { key: 'Irritability', label: 'Irritability', unit: '', placeholder: 'Yes/No' },
  { key: 'delayed_healing', label: 'Delayed Healing', unit: '', placeholder: 'Yes/No' },
  { key: 'partial_paresis', label: 'Partial Paresis', unit: '', placeholder: 'Yes/No' },
  { key: 'muscle_stiffness', label: 'Muscle Stiffness', unit: '', placeholder: 'Yes/No' },
  { key: 'Alopecia', label: 'Alopecia', unit: '', placeholder: 'Yes/No' },
  { key: 'Obesity', label: 'Obesity', unit: '', placeholder: 'Yes/No' },
]

const yesNoFields = new Set(['Gender', 'Polyuria', 'Polydipsia', 'sudden_weight_loss', 'weakness', 'Polyphagia', 'Genital_thrush', 'visual_blurring', 'Itching', 'Irritability', 'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'Alopecia', 'Obesity'])

interface Result { prediction: number; probability: number; risk_level: string; model_version: string }

function Page() {
  const [form, setForm] = useState<Record<string, string>>(Object.fromEntries(FIELDS.map(f => [f.key, ''])))
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(''); setLoading(true)
    try {
      const res = await fetch(`${API}/predict_male`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
        body: JSON.stringify(form),
      })
      if (!res.ok) throw new Error()
      setResult(await res.json())
    } catch { setError('Analysis failed. Please check inputs and try again.') }
    finally { setLoading(false) }
  }

  const selectStyle = { width: '100%', padding: '10px 14px', borderRadius: 10, border: '1.5px solid #e4e4e7', background: '#fafafa', fontSize: 14, color: '#09090b', transition: 'all 0.15s', appearance: 'none' as const, cursor: 'pointer' }

  return (
    <div style={{ minHeight: '100vh', background: '#fafafa' }}>
      <Navbar />
      <div style={{ maxWidth: 900, margin: '0 auto', padding: '48px 24px 80px' }}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 32 }}>
          <div style={{ width: 48, height: 48, borderRadius: 14, background: '#eff6ff', border: '1px solid #93c5fd', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><SyringeIcon size={24} color="#2563eb" strokeWidth={1.7} /></div>
          <div>
            <p style={{ fontSize: 12, color: '#0F9D9A', fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase' }}>Endocrinology</p>
            <h1 style={{ fontSize: 26, fontWeight: 800, color: '#09090b', letterSpacing: '-0.5px' }}>Male Diabetes Risk Assessment</h1>
          </div>
        </motion.div>

        {!result ? (
          <motion.form onSubmit={submit} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <div style={{ background: 'white', borderRadius: 20, border: '1px solid #f0f0f0', boxShadow: '0 2px 16px rgba(0,0,0,0.04)', overflow: 'hidden' }}>
              <div style={{ padding: '20px 28px', borderBottom: '1px solid #f4f4f5', background: 'linear-gradient(135deg, #eff6ff, #fff)' }}>
                <h2 style={{ fontWeight: 700, fontSize: 16, color: '#09090b' }}>Clinical Symptoms & Profile</h2>
                <p style={{ fontSize: 13, color: '#71717a', marginTop: 4 }}>16 symptom indicators for male diabetes screening</p>
              </div>
              <div style={{ padding: 28, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 20 }}>
                {FIELDS.map(f => (
                  <div key={f.key}>
                    <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#52525b', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>{f.label}</label>
                    {f.key === 'Age' ? (
                      <input type="number" required value={form[f.key]} onChange={e => setForm({ ...form, [f.key]: e.target.value })} placeholder={f.placeholder}
                        style={{ ...selectStyle, paddingRight: 52 }}
                        onFocus={e => { e.target.style.borderColor = '#0F9D9A'; e.target.style.background = 'white'; e.target.style.boxShadow = '0 0 0 3px rgba(15,157,154,0.12)' }}
                        onBlur={e => { e.target.style.borderColor = '#e4e4e7'; e.target.style.background = '#fafafa'; e.target.style.boxShadow = 'none' }} />
                    ) : (
                      <select required value={form[f.key]} onChange={e => setForm({ ...form, [f.key]: e.target.value })}
                        style={selectStyle}
                        onFocus={e => { e.target.style.borderColor = '#0F9D9A'; e.target.style.background = 'white'; e.target.style.boxShadow = '0 0 0 3px rgba(15,157,154,0.12)' }}
                        onBlur={e => { e.target.style.borderColor = '#e4e4e7'; e.target.style.background = '#fafafa'; e.target.style.boxShadow = 'none' }}>
                        <option value="">Select…</option>
                        {f.key === 'Gender' ? <><option value="Male">Male</option><option value="Female">Female</option></> : <><option value="Yes">Yes</option><option value="No">No</option></>}
                      </select>
                    )}
                  </div>
                ))}
              </div>
              {error && <div style={{ margin: '0 28px 20px', padding: '12px 16px', background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 10, color: '#dc2626', fontSize: 14 }}>{error}</div>}
              <div style={{ padding: '20px 28px', borderTop: '1px solid #f4f4f5' }}>
                <motion.button type="submit" disabled={loading} style={{ padding: '13px 32px', borderRadius: 12, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 16px rgba(15,157,154,0.3)' }}
                  whileHover={!loading ? { y: -2 } : {}} whileTap={{ scale: 0.97 }}>
                  {loading ? 'Analyzing…' : 'Run Analysis →'}
                </motion.button>
              </div>
            </div>
          </motion.form>
        ) : (
          <ResultDisplay result={result} onReset={() => setResult(null)} label={result.prediction === 1 ? 'Diabetes Risk Detected' : 'Low Diabetes Risk'} positiveColor="#2563eb" negativeColor="#16a34a" />
        )}
      </div>
    </div>
  )
}

export default function DiabetesMalePage() { return <RequireAuth><Page /></RequireAuth> }
