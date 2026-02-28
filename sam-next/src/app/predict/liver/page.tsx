'use client'
import { useState } from 'react'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { RequireAuth } from '@/components/AuthProvider'
import { ResultDisplay } from '../heart/page'
import { LiverIcon } from '@/components/MedicalIcons'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

type Option = { label: string; value: string }
type Field = { key: string; label: string; unit?: string; placeholder?: string; options?: Option[] }

const FIELDS: Field[] = [
  { key: 'Age', label: 'Age', unit: 'yrs', placeholder: 'Years' },
  { key: 'Total_Bilirubin', label: 'Total Bilirubin', unit: 'mg/dL', placeholder: '0.4–17.1' },
  { key: 'Direct_Bilirubin', label: 'Direct Bilirubin', unit: 'mg/dL', placeholder: '0.1–7.1' },
  { key: 'Alkaline_Phosphotase', label: 'Alkaline Phosphotase', unit: 'IU/L', placeholder: '44–147' },
  { key: 'Alamine_Aminotransferase', label: 'ALT (SGPT)', unit: 'IU/L', placeholder: '7–56' },
  { key: 'Aspartate_Aminotransferase', label: 'AST (SGOT)', unit: 'IU/L', placeholder: '10–40' },
  { key: 'Total_Protiens', label: 'Total Proteins', unit: 'g/dL', placeholder: '6.3–8.2' },
  { key: 'Albumin', label: 'Albumin', unit: 'g/dL', placeholder: '3.5–5.0' },
  { key: 'Albumin_and_Globulin_Ratio', label: 'A/G Ratio', unit: '', placeholder: '0.8–2.0' },
  { key: 'Gender_Male', label: 'Gender', unit: '', placeholder: '', options: [{ label: 'Male', value: '1' }, { label: 'Female', value: '0' }] },
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
      const res = await fetch(`${API}/detect_liver`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
        body: JSON.stringify({ input_data: FIELDS.map(f => Number(form[f.key])) }),
      })
      if (!res.ok) throw new Error()
      setResult(await res.json())
    } catch { setError('Analysis failed. Please check inputs and try again.') }
    finally { setLoading(false) }
  }

  return (
    <div style={{ minHeight: '100vh', background: '#fafafa' }}>
      <Navbar />
      <div style={{ maxWidth: 900, margin: '0 auto', padding: '48px 24px 80px' }}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 32 }}>
          <div style={{ width: 48, height: 48, borderRadius: 14, background: '#fffbeb', border: '1.5px solid #fcd34d', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><LiverIcon size={24} color="#d97706" strokeWidth={1.8} /></div>
          <div>
            <p style={{ fontSize: 12, color: '#0F9D9A', fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase' }}>Hepatology</p>
            <h1 style={{ fontSize: 26, fontWeight: 800, color: '#09090b', letterSpacing: '-0.5px' }}>Liver Health Assessment</h1>
          </div>
        </motion.div>

        {!result ? (
          <motion.form onSubmit={submit} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <div style={{ background: 'white', borderRadius: 20, border: '1px solid #f0f0f0', boxShadow: '0 2px 16px rgba(0,0,0,0.04)', overflow: 'hidden' }}>
              <div style={{ padding: '20px 28px', borderBottom: '1px solid #f4f4f5', background: 'linear-gradient(135deg, #fffbeb, #fff)' }}>
                <h2 style={{ fontWeight: 700, fontSize: 16, color: '#09090b' }}>Hepatic Enzyme Panel</h2>
                <p style={{ fontSize: 13, color: '#71717a', marginTop: 4 }}>Enter liver function test values</p>
              </div>
              <div style={{ padding: 28, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 20 }}>
                {FIELDS.map(f => (
                  <div key={f.key}>
                    <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#52525b', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>{f.label}</label>
                    <div style={{ position: 'relative' }}>
                      {f.options ? (
                        <select
                          required value={form[f.key]}
                          onChange={e => setForm({ ...form, [f.key]: e.target.value })}
                          style={{ width: '100%', padding: '10px 14px', borderRadius: 10, border: '1.5px solid #e4e4e7', background: '#fafafa', fontSize: 14, color: '#09090b', transition: 'all 0.15s', cursor: 'pointer', appearance: 'none' }}
                          onFocus={e => { e.target.style.borderColor = '#0F9D9A'; e.target.style.background = 'white'; e.target.style.boxShadow = '0 0 0 3px rgba(15,157,154,0.12)' }}
                          onBlur={e => { e.target.style.borderColor = '#e4e4e7'; e.target.style.background = '#fafafa'; e.target.style.boxShadow = 'none' }}
                        >
                          <option value="">Select…</option>
                          {f.options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>
                      ) : (
                        <input type="number" step="any" required value={form[f.key]} onChange={e => setForm({ ...form, [f.key]: e.target.value })} placeholder={f.placeholder}
                          style={{ width: '100%', padding: '10px 14px', paddingRight: f.unit ? 56 : 14, borderRadius: 10, border: '1.5px solid #e4e4e7', background: '#fafafa', fontSize: 14, color: '#09090b', transition: 'all 0.15s' }}
                          onFocus={e => { e.target.style.borderColor = '#0F9D9A'; e.target.style.background = 'white'; e.target.style.boxShadow = '0 0 0 3px rgba(15,157,154,0.12)' }}
                          onBlur={e => { e.target.style.borderColor = '#e4e4e7'; e.target.style.background = '#fafafa'; e.target.style.boxShadow = 'none' }} />
                      )}
                      {f.unit && !f.options && <span style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', fontSize: 10, color: '#a1a1aa', fontWeight: 700 }}>{f.unit}</span>}
                    </div>
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
          <ResultDisplay result={result} onReset={() => setResult(null)} label={result.prediction === 1 ? 'Liver Disease Risk' : 'Healthy Liver'} positiveColor="#b45309" negativeColor="#16a34a" />
        )}
      </div>
    </div>
  )
}

export default function LiverPage() { return <RequireAuth><Page /></RequireAuth> }
