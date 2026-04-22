'use client'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { RequireAuth } from '@/components/AuthProvider'
import { HeartECGIcon, DownloadIcon, RefreshIcon, AlertTriangleIcon, CheckCircleIcon, SparklesIcon, ChevronUpIcon, ChevronDownIcon } from '@/components/MedicalIcons'
import { saveAssessment } from '@/lib/assessmentHistory'
import { getTooltip } from '@/lib/fieldTooltips'
import { validateForm } from '@/lib/fieldValidation'
import FieldTooltip from '@/components/FieldTooltip'
import RecommendationsPanel from '@/components/RecommendationsPanel'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

type Option = { label: string; value: string }
type Field = { key: string; label: string; placeholder?: string; unit?: string; options?: Option[]; min?: number; max?: number }

const FIELDS: Field[] = [
  { key: 'age',      label: 'Age',                      placeholder: 'Years',  unit: 'yrs',   min: 1,   max: 120 },
  { key: 'sex',      label: 'Sex',                      options: [{ label: 'Male', value: '1' }, { label: 'Female', value: '0' }] },
  { key: 'cp',       label: 'Chest Pain Type',          placeholder: '0-3',                   min: 0,   max: 3 },
  { key: 'trestbps', label: 'Resting BP',               placeholder: 'mm Hg', unit: 'mmHg',  min: 60,  max: 250 },
  { key: 'chol',     label: 'Cholesterol',              placeholder: 'mg/dL', unit: 'mg/dL', min: 100, max: 700 },
  { key: 'fbs',      label: 'Fasting Blood Sugar >120', options: [{ label: 'Yes (>120 mg/dL)', value: '1' }, { label: 'No', value: '0' }] },
  { key: 'restecg',  label: 'Resting ECG',              placeholder: '0-2',                   min: 0,   max: 2 },
  { key: 'thalach',  label: 'Max Heart Rate',           placeholder: 'bpm',   unit: 'bpm',   min: 40,  max: 250 },
  { key: 'exang',    label: 'Exercise Angina',          options: [{ label: 'Yes', value: '1' }, { label: 'No', value: '0' }] },
  { key: 'oldpeak',  label: 'ST Depression',            placeholder: '0.0-6.2',               min: 0,   max: 10 },
  { key: 'slope',    label: 'ST Slope',                 placeholder: '0-2',                   min: 0,   max: 2 },
  { key: 'ca',       label: 'Major Vessels',            placeholder: '0-4',                   min: 0,   max: 4 },
  { key: 'thal',     label: 'Thalassemia',              placeholder: '1-3',                   min: 1,   max: 3 },
]

export interface Result { prediction: number; probability: number; risk_level: string; model_version: string }
export interface FormField { label: string; value: string; unit?: string }

export const DS = {
  page:     { minHeight: '100vh', background: '#0a0f1e' },
  wrap:     { maxWidth: 900, margin: '0 auto', padding: '48px 24px 80px' },
  card:     { background: '#111827', borderRadius: 22, border: '1px solid rgba(255,255,255,0.08)', boxShadow: '0 4px 24px rgba(0,0,0,0.30)', overflow: 'hidden' },
  cardHead: { padding: '20px 28px', borderBottom: '1px solid rgba(255,255,255,0.06)' },
  label:    { display: 'block' as const, fontSize: 12, fontWeight: 700, color: '#64748b', marginBottom: 6, textTransform: 'uppercase' as const, letterSpacing: '0.5px' },
  input:    { width: '100%', padding: '10px 14px', borderRadius: 10, border: '1.5px solid rgba(255,255,255,0.10)', background: 'rgba(255,255,255,0.04)', fontSize: 14, color: '#f1f5f9', transition: 'all 0.15s' },
  select:   { width: '100%', padding: '10px 14px', borderRadius: 10, border: '1.5px solid rgba(255,255,255,0.10)', background: '#1a2236', fontSize: 14, color: '#f1f5f9', transition: 'all 0.15s', cursor: 'pointer', appearance: 'none' as const },
}

async function downloadMedicalPDF(opts: {
  result: Result; assessmentType: string; formFields?: FormField[]; analysis: string
}) {
  const { default: jsPDF } = await import('jspdf')
  const { default: autoTable } = await import('jspdf-autotable')
  const { result, assessmentType, formFields = [], analysis } = opts
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
  const W = 210, M = 16, CW = W - M * 2
  const TEAL: [number, number, number] = [13, 148, 136]
  const RISK: Record<string, [number, number, number]> = {
    Low: [22, 163, 74], Moderate: [217, 119, 6], High: [220, 38, 38],
  }
  const rRgb = RISK[result.risk_level] ?? RISK.Low

  // ── Header ────────────────────────────────────────────────
  doc.setFillColor(...TEAL)
  doc.rect(0, 0, W, 38, 'F')
  doc.setTextColor(255, 255, 255)
  doc.setFontSize(20)
  doc.setFont('helvetica', 'bold')
  doc.text('SAM AI', M, 17)
  doc.setFontSize(8)
  doc.setFont('helvetica', 'normal')
  doc.text('Clinical Assessment Report', M, 26)
  doc.setTextColor(180, 235, 230)
  doc.text(new Date().toLocaleString(), W - M, 17, { align: 'right' })
  doc.text(assessmentType + ' Assessment', W - M, 26, { align: 'right' })

  let y = 46

  // ── Result summary box ────────────────────────────────────
  doc.setFillColor(249, 250, 251)
  doc.roundedRect(M, y, CW, 50, 3, 3, 'F')
  doc.setFillColor(...rRgb)
  doc.rect(M, y, 4, 50, 'F')
  doc.roundedRect(M, y, 4, 50, 2, 2, 'F')
  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...rRgb)
  doc.text(result.prediction === 1 ? 'Condition Detected' : 'No Condition Detected', M + 11, y + 14)
  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(75, 85, 99)
  doc.text(assessmentType, M + 11, y + 23)
  doc.setFillColor(...rRgb)
  doc.roundedRect(M + 11, y + 28, 26, 7, 2, 2, 'F')
  doc.setFontSize(7)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(255, 255, 255)
  doc.text(result.risk_level + ' Risk', M + 24, y + 33, { align: 'center' })
  doc.setFontSize(8)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(107, 114, 128)
  doc.text('Confidence Score', W - M - 8, y + 10, { align: 'right' })
  doc.setFontSize(22)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...rRgb)
  doc.text(result.probability.toFixed(1) + '%', W - M - 8, y + 24, { align: 'right' })
  const bw = 56, bx = W - M - 8 - bw
  doc.setFillColor(209, 213, 219)
  doc.roundedRect(bx, y + 29, bw, 4, 2, 2, 'F')
  doc.setFillColor(...rRgb)
  doc.roundedRect(bx, y + 29, Math.max(2, (result.probability / 100) * bw), 4, 2, 2, 'F')
  doc.setFontSize(7)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(156, 163, 175)
  doc.text('Model v' + (result.model_version || '1.0'), W - M - 8, y + 41, { align: 'right' })
  y += 58

  // ── Clinical data table ───────────────────────────────────
  doc.setFontSize(7)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(107, 114, 128)
  doc.text('CLINICAL INPUT DATA', M, y)
  y += 4
  autoTable(doc, {
    startY: y,
    head: [['Parameter', 'Value', 'Unit']],
    body: formFields.filter(f => f.value).map(f => [f.label, f.value, f.unit || '-']),
    theme: 'grid',
    styles: { fontSize: 8.5, cellPadding: 3.2, textColor: [30, 41, 59] as [number,number,number], lineColor: [226, 232, 240] as [number,number,number], lineWidth: 0.25 },
    headStyles: { fillColor: TEAL, textColor: [255, 255, 255] as [number,number,number], fontStyle: 'bold', fontSize: 8.5 },
    alternateRowStyles: { fillColor: [249, 250, 251] as [number,number,number] },
    bodyStyles: { fillColor: [255, 255, 255] as [number,number,number] },
    columnStyles: {
      0: { fontStyle: 'bold', textColor: [15, 23, 42] as [number,number,number], cellWidth: 75 },
      1: { textColor: [30, 41, 59] as [number,number,number], halign: 'center' as const, cellWidth: 50 },
      2: { textColor: [100, 116, 139] as [number,number,number], halign: 'center' as const, fontSize: 7.5 },
    },
    margin: { left: M, right: M },
  })
  y = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 10

  // ── AI analysis ───────────────────────────────────────────
  if (analysis && analysis.trim()) {
    if (y > 218) { doc.addPage(); y = 18 }
    doc.setFontSize(7)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(107, 114, 128)
    doc.text('AI MEDICAL ANALYSIS', M, y)
    y += 5
    doc.setFontSize(7)
    doc.setFont('helvetica', 'italic')
    doc.setTextColor(156, 163, 175)
    doc.text('Powered by Groq AI - llama-3.3-70b-versatile - For educational purposes only', M, y)
    y += 6
    const paragraphs = analysis.split('\n\n').filter((p: string) => p.trim())
    for (const para of paragraphs) {
      const lines = doc.splitTextToSize(para.trim(), CW - 10)
      const bh = lines.length * 4.6 + 6
      if (y + bh > 272) { doc.addPage(); y = 18 }
      doc.setFillColor(240, 253, 250)
      doc.roundedRect(M, y - 2, CW, lines.length * 4.6 + 6, 2, 2, 'F')
      doc.setFillColor(...TEAL)
      doc.rect(M, y - 2, 3, lines.length * 4.6 + 6, 'F')
      doc.roundedRect(M, y - 2, 3, lines.length * 4.6 + 6, 2, 2, 'F')
      doc.setFontSize(9)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(30, 41, 59)
      doc.text(lines, M + 7, y + 2.5)
      y += bh + 2
    }
    y += 4
  }

  // ── Disclaimer ────────────────────────────────────────────
  if (y > 258) { doc.addPage(); y = 18 }
  const discText = 'SAM AI is for educational purposes only. This report does not constitute a medical diagnosis or treatment recommendation. AI predictions are based on statistical models and may not be accurate for every individual. Always consult a qualified healthcare professional before making any medical decisions.'
  const discLines = doc.splitTextToSize(discText, CW - 12)
  const discH = discLines.length * 4.2 + 16
  doc.setFillColor(255, 251, 235)
  doc.roundedRect(M, y, CW, discH, 3, 3, 'F')
  doc.setFillColor(245, 158, 11)
  doc.rect(M, y, 3, discH, 'F')
  doc.roundedRect(M, y, 3, discH, 2, 2, 'F')
  doc.setFontSize(7)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(146, 64, 14)
  doc.text('MEDICAL DISCLAIMER', M + 7, y + 8)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(92, 67, 26)
  doc.text(discLines, M + 7, y + 14)

  // ── Footer ────────────────────────────────────────────────
  const totalPg = (doc as unknown as { getNumberOfPages: () => number }).getNumberOfPages()
  for (let i = 1; i <= totalPg; i++) {
    doc.setPage(i)
    doc.setDrawColor(226, 232, 240)
    doc.setLineWidth(0.3)
    doc.line(M, 285, W - M, 285)
    doc.setFontSize(7)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(156, 163, 175)
    doc.text('Page ' + i + ' of ' + totalPg, W / 2, 291, { align: 'center' })
    doc.text('SAM AI - Clinical Assessment System', M, 291)
    doc.text(new Date().toLocaleDateString(), W - M, 291, { align: 'right' })
  }

  doc.save('SAM-AI-' + assessmentType.replace(/[\s/()+]+/g, '-') + '-Report.pdf')
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
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
        body: JSON.stringify({ data: Object.fromEntries(Object.entries(form).map(([k, v]) => [k, Number(v)])) }),
      })
      if (!res.ok) throw new Error('Prediction failed')
      const data = await res.json()
      setResult(data)
      saveAssessment({ type: 'heart', title: 'Heart Disease', prediction: data.prediction, probability: data.probability, risk_level: data.risk_level })
    } catch { setError('Analysis failed. Please check your inputs and try again.') }
    finally { setLoading(false) }
  }

  const resolvedFormFields: FormField[] = FIELDS.map(f => ({
    label: f.label,
    value: f.options ? (f.options.find(o => o.value === form[f.key])?.label ?? form[f.key] ?? '') : (form[f.key] ?? ''),
    unit: f.unit,
  }))

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
                <p style={{ fontSize: 13, color: '#64748b', marginTop: 3 }}>Fill in all 13 cardiovascular markers.</p>
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
                          style={{ ...DS.select, ...(errors[f.key] ? { borderColor: 'rgba(248,113,113,0.60)' } : {}) }}>
                          <option value="">Select...</option>
                          {f.options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>
                      ) : (
                        <input type="number" step="any" value={form[f.key]} onChange={e => setField(f.key, e.target.value)} placeholder={f.placeholder}
                          style={{ ...DS.input, paddingRight: f.unit ? 52 : 14, ...(errors[f.key] ? { borderColor: 'rgba(248,113,113,0.60)' } : {}) }} />
                      )}
                      {f.unit && !f.options && <span style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', fontSize: 10, color: '#475569', fontWeight: 700 }}>{f.unit}</span>}
                    </div>
                    {errors[f.key] && <p style={{ fontSize: 11, color: '#f87171', marginTop: 5 }}>⚠ {errors[f.key]}</p>}
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
                  whileHover={!loading ? { y: -2 } : {}} whileTap={{ scale: 0.97 }}>
                  <HeartECGIcon size={16} color="white" strokeWidth={2} />
                  {loading ? 'Analysing...' : 'Run Analysis'}
                </motion.button>
              </div>
            </div>
          </motion.form>
        ) : (
          <div>
            <ResultDisplay result={result} onReset={() => setResult(null)}
              label={result.prediction === 1 ? 'Disease Detected' : 'No Disease Detected'}
              positiveColor="#f87171" negativeColor="#34d399" assessmentType="Heart Disease" />
            <RecommendationsPanel type="heart" risk_level={result.risk_level} />
            <ExportReport result={result} assessmentType="Heart Disease" formFields={resolvedFormFields} />
          </div>
        )}
      </div>
    </div>
  )
}

export function ResultDisplay({ result, onReset, label, positiveColor, negativeColor, assessmentType }: {
  result: Result; onReset: () => void; label: string; positiveColor: string; negativeColor: string; assessmentType?: string
}) {
  const isPositive = result.prediction === 1
  const color = isPositive ? positiveColor : negativeColor
  const riskColor  = result.risk_level === 'Low' ? '#34d399' : result.risk_level === 'Moderate' ? '#fbbf24' : '#f87171'
  const riskBg     = result.risk_level === 'Low' ? 'rgba(52,211,153,0.10)' : result.risk_level === 'Moderate' ? 'rgba(251,191,36,0.10)' : 'rgba(248,113,113,0.10)'
  const riskBorder = result.risk_level === 'Low' ? 'rgba(52,211,153,0.25)' : result.risk_level === 'Moderate' ? 'rgba(251,191,36,0.25)' : 'rgba(248,113,113,0.25)'
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      style={{ background: '#111827', borderRadius: 22, border: '1px solid rgba(255,255,255,0.08)', boxShadow: '0 8px 40px rgba(0,0,0,0.30)', overflow: 'hidden' }}>
      <div style={{ padding: '28px 32px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            {isPositive ? <AlertTriangleIcon size={18} color={positiveColor} strokeWidth={2} /> : <CheckCircleIcon size={18} color={negativeColor} strokeWidth={2} />}
            <p style={{ fontSize: 11, color: '#64748b', fontWeight: 600, letterSpacing: 1, textTransform: 'uppercase' }}>Analysis Result</p>
          </div>
          <h2 style={{ fontSize: 28, fontWeight: 800, color, letterSpacing: '-0.5px' }}>{label}</h2>
        </div>
        <div style={{ padding: '8px 20px', borderRadius: 100, background: riskBg, color: riskColor, fontWeight: 700, fontSize: 14, border: '1px solid ' + riskBorder }}>
          {result.risk_level} Risk
        </div>
      </div>
      <div style={{ padding: '32px' }}>
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
            <span style={{ fontSize: 13, color: '#64748b', fontWeight: 600 }}>Model Confidence Score</span>
            <span style={{ fontSize: 16, fontWeight: 800, color: '#f1f5f9' }}>{result.probability.toFixed(1)}%</span>
          </div>
          <div style={{ height: 10, background: 'rgba(255,255,255,0.06)', borderRadius: 100, overflow: 'hidden' }}>
            <motion.div style={{ height: '100%', borderRadius: 100, background: 'linear-gradient(90deg, ' + color + '60, ' + color + ')' }}
              initial={{ width: 0 }} animate={{ width: result.probability + '%' }}
              transition={{ duration: 1.2, ease: 'easeOut', delay: 0.2 }} />
          </div>
        </div>
        <p style={{ fontSize: 13, color: '#475569', marginBottom: 28 }}>
          {result.model_version && 'Model v' + result.model_version + ' · '}For educational purposes only
        </p>
        <motion.button onClick={onReset}
          style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '11px 24px', borderRadius: 11, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.10)', color: '#94a3b8', fontWeight: 600, fontSize: 14, cursor: 'pointer' }}
          whileHover={{ y: -1, background: 'rgba(255,255,255,0.08)' as string }} whileTap={{ scale: 0.98 }}>
          <RefreshIcon size={14} color="#94a3b8" strokeWidth={2.2} />
          New Assessment
        </motion.button>
      </div>
    </motion.div>
  )
}

export function ExportReport({ result, assessmentType, formFields = [] }: {
  result: Result; assessmentType: string; formFields?: FormField[]
}) {
  const [status, setStatus] = useState<'idle' | 'loading' | 'done'>('idle')
  const [aiInsights, setAiInsights] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(true)

  const handleExport = async () => {
    if (status === 'loading') return
    setStatus('loading')
    setAiInsights(null)
    let analysis = ''
    try {
      const aiRes = await fetch('/api/generate-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ assessmentType, prediction: result.prediction, probability: result.probability, risk_level: result.risk_level, formFields }),
      })
      if (aiRes.ok) {
        const d = await aiRes.json()
        analysis = d.analysis ?? ''
        if (analysis) setAiInsights(analysis)
      }
    } catch { /* generate PDF without AI */ }
    try { await downloadMedicalPDF({ result, assessmentType, formFields, analysis }) }
    catch (err) { console.error('PDF error:', err) }
    setStatus('done')
    setExpanded(true)
  }

  const btnColor = status === 'done' ? '#34d399' : '#14b8a6'
  const btnBg    = status === 'done' ? 'rgba(52,211,153,0.08)' : 'rgba(20,184,166,0.08)'
  const btnBdr   = status === 'done' ? 'rgba(52,211,153,0.25)' : 'rgba(20,184,166,0.25)'

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} style={{ marginTop: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <motion.button onClick={handleExport} disabled={status === 'loading'}
          style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '10px 20px', borderRadius: 10, background: btnBg, border: '1px solid ' + btnBdr, color: btnColor, fontWeight: 600, fontSize: 13, cursor: status === 'loading' ? 'not-allowed' : 'pointer', opacity: status === 'loading' ? 0.7 : 1, transition: 'all 0.15s' }}
          whileHover={status !== 'loading' ? { y: -1 } : {}} whileTap={{ scale: 0.97 }}>
          {status === 'loading' ? (
            <>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" style={{ animation: 'spin 1s linear infinite' }}>
                <path d="M21 12a9 9 0 11-6.219-8.56"/>
              </svg>
              Generating report...
            </>
          ) : status === 'done' ? (
            <>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12"/></svg>
              Downloaded - Export Again
            </>
          ) : (
            <>
              <DownloadIcon size={13} color="currentColor" strokeWidth={2.2} />
              Export Report
            </>
          )}
        </motion.button>
      </div>
      <AnimatePresence>
        {aiInsights && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            style={{ marginTop: 14, background: 'linear-gradient(135deg, rgba(20,184,166,0.06), rgba(96,165,250,0.04))', border: '1px solid rgba(20,184,166,0.20)', borderRadius: 16, padding: '18px 22px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: expanded ? 14 : 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ width: 30, height: 30, borderRadius: 9, background: 'rgba(20,184,166,0.12)', border: '1px solid rgba(20,184,166,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <SparklesIcon size={14} color="#14b8a6" strokeWidth={2} />
                </div>
                <div>
                  <p style={{ fontWeight: 700, fontSize: 13, color: '#f1f5f9' }}>AI Medical Insights</p>
                  <p style={{ fontSize: 10, color: '#475569' }}>Groq · llama-3.3-70b · Included in your PDF</p>
                </div>
              </div>
              <button onClick={() => setExpanded(v => !v)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#475569', padding: 4, display: 'flex' }}>
                {expanded ? <ChevronUpIcon size={15} color="#475569" /> : <ChevronDownIcon size={15} color="#475569" />}
              </button>
            </div>
            <AnimatePresence initial={false}>
              {expanded && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} style={{ overflow: 'hidden' }}>
                  {aiInsights.split('\n\n').filter(p => p.trim()).map((para, i, arr) => (
                    <p key={i} style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.75, marginBottom: i < arr.length - 1 ? 12 : 0 }}>
                      {para.trim()}
                    </p>
                  ))}
                  <p style={{ fontSize: 11, color: '#334155', marginTop: 12, paddingTop: 10, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                    Educational purposes only — not a medical diagnosis. Consult a qualified healthcare professional.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export const PrintReport = ExportReport
export { DS as DARK_STYLES }
export default function HeartPage() { return <RequireAuth><Page /></RequireAuth> }
