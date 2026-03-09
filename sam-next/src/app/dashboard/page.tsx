'use client'
import { useState, useEffect } from 'react'
import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { RequireAuth } from '@/components/AuthProvider'
import { HeartECGIcon, LiverIcon, GlucoseDropIcon, SyringeIcon, MicroscopeIcon, ActivityIcon, ArrowRightIcon, DownloadIcon, SparklesIcon, ShieldCheckIcon, TargetIcon } from '@/components/MedicalIcons'
import { getAssessments, deleteAssessment, clearAssessments, type Assessment, type AssessmentType } from '@/lib/assessmentHistory'
import RiskTrendChart from '@/components/RiskTrendChart'

const modules = [
  { title: 'Heart Disease', desc: 'Cardiovascular risk from 13 clinical vitals.', href: '/predict/heart', accent: 'rgba(248,113,113,0.12)', border: 'rgba(248,113,113,0.28)', iconColor: '#f87171', tag: 'Cardiology', Icon: HeartECGIcon },
  { title: 'Liver Health', desc: 'Hepatic enzyme panel — bilirubin, ALT, AST, albumin.', href: '/predict/liver', accent: 'rgba(251,191,36,0.12)', border: 'rgba(251,191,36,0.28)', iconColor: '#fbbf24', tag: 'Hepatology', Icon: LiverIcon },
  { title: 'Diabetes (Female)', desc: 'Metabolic risk: glucose, BMI, insulin, pedigree function.', href: '/predict/diabetes-female', accent: 'rgba(167,139,250,0.12)', border: 'rgba(167,139,250,0.28)', iconColor: '#a78bfa', tag: 'Endocrinology', Icon: GlucoseDropIcon },
  { title: 'Diabetes (Male)', desc: 'Symptom screening: 16 clinical and lifestyle indicators.', href: '/predict/diabetes-male', accent: 'rgba(96,165,250,0.12)', border: 'rgba(96,165,250,0.28)', iconColor: '#60a5fa', tag: 'Endocrinology', Icon: SyringeIcon },
  { title: 'Breast Cancer', desc: '30 FNA cytology markers — malignant vs benign.', href: '/predict/breast-cancer', accent: 'rgba(52,211,153,0.12)', border: 'rgba(52,211,153,0.28)', iconColor: '#34d399', tag: 'Oncology', Icon: MicroscopeIcon },
]

const moduleMap: Record<AssessmentType, typeof modules[0]> = {
  'heart':           modules[0],
  'liver':           modules[1],
  'diabetes-female': modules[2],
  'diabetes-male':   modules[3],
  'breast-cancer':   modules[4],
}

const FILTERS: Array<{ label: string; value: AssessmentType | 'all' }> = [
  { label: 'All', value: 'all' },
  { label: 'Heart', value: 'heart' },
  { label: 'Liver', value: 'liver' },
  { label: 'Diabetes ♀', value: 'diabetes-female' },
  { label: 'Diabetes ♂', value: 'diabetes-male' },
  { label: 'Breast Cancer', value: 'breast-cancer' },
]

function formatDate(iso: string) {
  const d = new Date(iso)
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) + ' · ' +
    d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
}

function RiskBadge({ level }: { level: string }) {
  const c = level === 'Low' ? '#34d399' : level === 'Moderate' ? '#fbbf24' : '#f87171'
  const bg = level === 'Low' ? 'rgba(52,211,153,0.10)' : level === 'Moderate' ? 'rgba(251,191,36,0.10)' : 'rgba(248,113,113,0.10)'
  return <span style={{ fontSize: 11, fontWeight: 700, padding: '3px 10px', borderRadius: 100, background: bg, color: c, border: `1px solid ${c}30` }}>{level} Risk</span>
}

function AssessmentCard({ a, onDelete }: { a: Assessment; onDelete: () => void }) {
  const mod = moduleMap[a.type]
  const barColor = a.risk_level === 'Low' ? '#34d399' : a.risk_level === 'Moderate' ? '#fbbf24' : '#f87171'

  return (
    <motion.div layout initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.96 }}
      style={{ background: '#111827', borderRadius: 18, border: '1px solid rgba(255,255,255,0.07)', padding: '18px 20px' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1, minWidth: 0 }}>
          <div style={{ width: 40, height: 40, borderRadius: 12, background: mod.accent, border: `1.5px solid ${mod.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <mod.Icon size={19} color={mod.iconColor} strokeWidth={1.7} />
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 700, fontSize: 14, color: '#f1f5f9' }}>{a.title}</span>
              <RiskBadge level={a.risk_level} />
            </div>
            <p style={{ fontSize: 11, color: '#475569', marginTop: 3 }}>{formatDate(a.date)}</p>
          </div>
        </div>
        <button onClick={onDelete} aria-label="Delete"
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#334155', padding: 4, borderRadius: 6, display: 'flex', transition: 'color 0.15s', flexShrink: 0 }}
          onMouseEnter={e => (e.currentTarget.style.color = '#f87171')}
          onMouseLeave={e => (e.currentTarget.style.color = '#334155')}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/></svg>
        </button>
      </div>
      <div style={{ marginTop: 14 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontSize: 11, color: '#475569', fontWeight: 600 }}>{a.prediction === 1 ? 'Positive' : 'Negative'} · Confidence</span>
          <span style={{ fontSize: 12, fontWeight: 800, color: '#94a3b8' }}>{a.probability.toFixed(1)}%</span>
        </div>
        <div style={{ height: 5, background: 'rgba(255,255,255,0.05)', borderRadius: 100, overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${a.probability}%`, borderRadius: 100, background: `linear-gradient(90deg, ${barColor}60, ${barColor})`, transition: 'width 0.8s ease' }} />
        </div>
      </div>
    </motion.div>
  )
}

function HistorySection({ assessments, setAssessments }: { assessments: Assessment[]; setAssessments: React.Dispatch<React.SetStateAction<Assessment[]>> }) {
  const [filter, setFilter] = useState<AssessmentType | 'all'>('all')

  const filtered = filter === 'all' ? assessments : assessments.filter(a => a.type === filter)

  const handleDelete = (id: string) => {
    deleteAssessment(id)
    setAssessments(prev => prev.filter(a => a.id !== id))
  }

  const handleClear = () => { clearAssessments(); setAssessments([]) }

  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(assessments, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'sam-ai-assessments.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} style={{ marginBottom: 40 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 18 }}>
        <div>
          <h2 style={{ fontSize: 19, fontWeight: 800, color: '#f1f5f9', letterSpacing: '-0.3px' }}>Past Assessments</h2>
          <p style={{ fontSize: 12, color: '#475569', marginTop: 3 }}>{assessments.length} {assessments.length === 1 ? 'record' : 'records'} stored locally</p>
        </div>
        {assessments.length > 0 && (
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <button onClick={handleDownload}
              style={{ fontSize: 12, fontWeight: 700, color: '#14b8a6', background: 'rgba(20,184,166,0.08)', border: '1px solid rgba(20,184,166,0.25)', borderRadius: 8, padding: '7px 12px', cursor: 'pointer', display: 'flex', gap: 6, alignItems: 'center' }}>
              <DownloadIcon size={13} color="#14b8a6" strokeWidth={2} /> Export
            </button>
            <button onClick={handleClear}
              style={{ fontSize: 12, fontWeight: 600, color: '#475569', background: 'none', border: '1px solid rgba(255,255,255,0.09)', borderRadius: 8, padding: '6px 12px', cursor: 'pointer', transition: 'all 0.15s' }}
              onMouseEnter={e => { e.currentTarget.style.color = '#f87171'; e.currentTarget.style.borderColor = 'rgba(248,113,113,0.30)' }}
              onMouseLeave={e => { e.currentTarget.style.color = '#475569'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.09)' }}>
              Clear all
            </button>
          </div>
        )}
      </div>

      {assessments.length > 0 && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 18 }}>
          {FILTERS.map(opt => (
            <button key={opt.value} onClick={() => setFilter(opt.value)}
              style={{ fontSize: 12, fontWeight: 600, padding: '5px 14px', borderRadius: 100, border: '1.5px solid', cursor: 'pointer', transition: 'all 0.15s',
                background: filter === opt.value ? '#14b8a6' : 'rgba(255,255,255,0.04)',
                color: filter === opt.value ? 'white' : '#64748b',
                borderColor: filter === opt.value ? '#14b8a6' : 'rgba(255,255,255,0.09)',
              }}>
              {opt.label}
            </button>
          ))}
        </div>
      )}

      {assessments.length === 0 ? (
        <div style={{ background: '#111827', border: '1.5px dashed rgba(255,255,255,0.09)', borderRadius: 18, padding: '40px 24px', textAlign: 'center' }}>
          <div style={{ width: 48, height: 48, borderRadius: 14, background: 'rgba(255,255,255,0.04)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 14px' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="1.7" strokeLinecap="round"><path d="M9 12h6M9 16h6M17 21H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z"/></svg>
          </div>
          <p style={{ fontWeight: 700, color: '#64748b', fontSize: 14, marginBottom: 5 }}>No assessments yet</p>
          <p style={{ fontSize: 12, color: '#334155' }}>Complete a diagnostic module below to start building your history.</p>
        </div>
      ) : filtered.length === 0 ? (
        <p style={{ fontSize: 13, color: '#475569', padding: '20px 0' }}>No assessments match this filter.</p>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 14 }}>
          <AnimatePresence>
            {filtered.map(a => <AssessmentCard key={a.id} a={a} onDelete={() => handleDelete(a.id)} />)}
          </AnimatePresence>
        </div>
      )}
    </motion.section>
  )
}

function DashboardContent() {
  const [assessments, setAssessments] = useState<Assessment[]>([])
  useEffect(() => { setAssessments(getAssessments()) }, [])

  const latest = assessments[0]
  const avgConfidence = assessments.length ? assessments.reduce((sum, a) => sum + a.probability, 0) / assessments.length : 0
  const highRiskCount = assessments.filter(a => a.risk_level === 'High').length
  const moduleCoverage = new Set(assessments.map(a => a.type)).size

  return (
    <div style={{ minHeight: '100vh', background: '#0a0f1e' }}>
      <Navbar />
      <main style={{ maxWidth: 1100, margin: '0 auto', padding: '48px 24px 80px' }}>

        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} style={{ marginBottom: 40 }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
            <div>
              <p style={{ fontSize: 11, fontWeight: 700, color: '#14b8a6', letterSpacing: 2, marginBottom: 6, textTransform: 'uppercase' }}>Clinical Dashboard</p>
              <h1 style={{ fontSize: 32, fontWeight: 800, color: '#f1f5f9', letterSpacing: '-0.8px' }}>AI Diagnostic Suite</h1>
              <p style={{ color: '#64748b', fontSize: 15, marginTop: 6 }}>Your personalised clinical assessment hub</p>
            </div>
            <div style={{ background: 'linear-gradient(135deg, rgba(20,184,166,0.14), rgba(96,165,250,0.14))', border: '1px solid rgba(20,184,166,0.30)', borderRadius: 14, padding: '12px 20px', display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#34d399', animation: 'pulse-dot 2s infinite', boxShadow: '0 0 12px rgba(52,211,153,0.7)' }} />
              <span style={{ fontSize: 13, fontWeight: 700, color: '#e0f2fe' }}>5 modules live · Real-time inference</span>
            </div>
          </div>
        </motion.div>

        {/* Summary cards */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14, marginBottom: 28 }}>
          <div style={{ background: '#111827', borderRadius: 16, border: '1px solid rgba(20,184,166,0.25)', padding: '16px 18px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 38, height: 38, borderRadius: 10, background: 'rgba(20,184,166,0.14)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <SparklesIcon size={18} color="#14b8a6" strokeWidth={2.2} />
            </div>
            <div>
              <p style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Latest result</p>
              <p style={{ fontSize: 15, fontWeight: 800, color: '#f1f5f9' }}>{latest ? `${latest.title} · ${latest.risk_level} Risk` : 'Awaiting first run'}</p>
            </div>
          </div>
          <div style={{ background: '#111827', borderRadius: 16, border: '1px solid rgba(255,255,255,0.08)', padding: '16px 18px' }}>
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Average confidence</p>
            <p style={{ fontSize: 28, fontWeight: 800, color: '#e0f2fe' }}>{avgConfidence.toFixed(1)}%</p>
            <p style={{ fontSize: 11, color: '#334155', marginTop: 2 }}>Across all saved assessments</p>
          </div>
          <div style={{ background: '#111827', borderRadius: 16, border: '1px solid rgba(248,113,113,0.18)', padding: '16px 18px' }}>
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>High-risk flags</p>
            <p style={{ fontSize: 28, fontWeight: 800, color: '#f87171' }}>{highRiskCount}</p>
            <p style={{ fontSize: 11, color: '#334155', marginTop: 2 }}>Past results tagged as High</p>
          </div>
          <div style={{ background: '#111827', borderRadius: 16, border: '1px solid rgba(96,165,250,0.20)', padding: '16px 18px' }}>
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Module coverage</p>
            <p style={{ fontSize: 28, fontWeight: 800, color: '#60a5fa' }}>{moduleCoverage}/5</p>
            <p style={{ fontSize: 11, color: '#334155', marginTop: 2 }}>Unique specialties assessed</p>
          </div>
        </motion.div>

        {/* Next best action */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 }} style={{ background: 'linear-gradient(120deg, rgba(20,184,166,0.08), rgba(52,211,153,0.08))', border: '1px solid rgba(20,184,166,0.15)', borderRadius: 18, padding: '18px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap', marginBottom: 28 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 42, height: 42, borderRadius: 12, background: 'rgba(20,184,166,0.18)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <TargetIcon size={20} color="#14b8a6" strokeWidth={2.2} />
            </div>
            <div>
              <p style={{ fontSize: 12, color: '#14b8a6', fontWeight: 700, letterSpacing: 0.4 }}>Next Best Action</p>
              <p style={{ fontSize: 15, fontWeight: 800, color: '#f1f5f9' }}>{latest ? `Run a follow-up ${latest.title} assessment to track changes.` : 'Run your first assessment to unlock trends and insights.'}</p>
              <p style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>Keep assessments consistent to compare risk trajectories.</p>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Link href={latest ? moduleMap[latest.type].href : '/predict/heart'} style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '10px 14px', borderRadius: 10, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontWeight: 700, fontSize: 13, textDecoration: 'none', boxShadow: '0 10px 30px rgba(20,184,166,0.35)' }}>
              Resume module
              <ArrowRightIcon size={13} color="white" strokeWidth={2.4} />
            </Link>
            <Link href="/predict/heart" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '10px 14px', borderRadius: 10, border: '1px solid rgba(255,255,255,0.14)', color: '#cbd5e1', fontWeight: 700, fontSize: 13, textDecoration: 'none', background: '#111827' }}>
              See all modules
            </Link>
          </div>
        </motion.div>

        {/* Risk Trend Chart */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} style={{ marginBottom: 36 }}>
          <div style={{ marginBottom: 14 }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#f1f5f9' }}>Risk Trend</h2>
            <p style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>Confidence trajectory across all past assessments</p>
          </div>
          <RiskTrendChart assessments={assessments} />
        </motion.div>

        {/* History */}
        <HistorySection assessments={assessments} setAssessments={setAssessments} />

        {/* Module grid */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <div style={{ marginBottom: 20 }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#f1f5f9' }}>Run New Assessment</h2>
            <p style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>Select a module below to begin</p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16, marginBottom: 32 }}>
            {modules.map((m, i) => (
              <motion.div key={m.title} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 + i * 0.06 }}>
                <Link href={m.href} style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#111827', borderRadius: 20, padding: 24, textDecoration: 'none', border: `1px solid ${m.border}`, position: 'relative', overflow: 'hidden', transition: 'all 0.2s' }}
                  onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = `0 16px 40px rgba(0,0,0,0.30)` }}
                  onMouseLeave={e => { e.currentTarget.style.transform = ''; e.currentTarget.style.boxShadow = '' }}>
                  <div style={{ position: 'absolute', top: -20, right: -20, width: 100, height: 100, borderRadius: '50%', background: m.accent, filter: 'blur(30px)', pointerEvents: 'none' }} />
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 16, position: 'relative' }}>
                    <div style={{ width: 46, height: 46, borderRadius: 14, background: m.accent, border: `1.5px solid ${m.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <m.Icon size={22} color={m.iconColor} strokeWidth={1.7} />
                    </div>
                    <span style={{ fontSize: 10, fontWeight: 700, color: '#475569', background: 'rgba(255,255,255,0.05)', padding: '3px 9px', borderRadius: 100, border: '1px solid rgba(255,255,255,0.08)' }}>{m.tag}</span>
                  </div>
                  <h2 style={{ fontWeight: 700, fontSize: 15, color: '#f1f5f9', marginBottom: 6, position: 'relative' }}>{m.title}</h2>
                  <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.65, flex: 1, marginBottom: 16, position: 'relative' }}>{m.desc}</p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#14b8a6', fontSize: 12, fontWeight: 700, position: 'relative' }}>
                    Start Assessment <ArrowRightIcon size={12} color="#14b8a6" strokeWidth={2.5} />
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Disclaimer */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
          style={{ background: 'rgba(20,184,166,0.05)', border: '1px solid rgba(20,184,166,0.13)', borderRadius: 14, padding: '16px 22px', display: 'flex', alignItems: 'center', gap: 12 }}>
          <ActivityIcon size={15} color="#14b8a6" strokeWidth={2} style={{ flexShrink: 0 }} />
          <p style={{ fontSize: 13, color: '#64748b', lineHeight: 1.6 }}>
            <strong style={{ color: '#94a3b8' }}>Medical Disclaimer:</strong> SAM AI is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
          </p>
        </motion.div>
      </main>
    </div>
  )
}

export default function DashboardPage() { return <RequireAuth><DashboardContent /></RequireAuth> }
