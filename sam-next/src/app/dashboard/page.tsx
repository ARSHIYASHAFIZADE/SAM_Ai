'use client'
import Link from 'next/link'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { RequireAuth } from '@/components/AuthProvider'
import { HeartECGIcon, LiverIcon, GlucoseDropIcon, SyringeIcon, MicroscopeIcon } from '@/components/MedicalIcons'

const modules = [
  { title: 'Heart Disease', desc: 'Cardiovascular risk from 13 clinical vitals — age, cholesterol, ECG, and more.', href: '/predict/heart', accent: '#fef2f2', border: '#fca5a5', iconColor: '#ef4444', tag: 'Cardiology', Icon: HeartECGIcon },
  { title: 'Liver Health', desc: 'Hepatic enzyme analysis — bilirubin, ALT, AST, albumin, and globulin ratio.', href: '/predict/liver', accent: '#fffbeb', border: '#fcd34d', iconColor: '#d97706', tag: 'Hepatology', Icon: LiverIcon },
  { title: 'Diabetes (Female)', desc: 'Metabolic risk: glucose, BMI, insulin, and pedigree function from PIMA dataset.', href: '/predict/diabetes-female', accent: '#fdf4ff', border: '#d8b4fe', iconColor: '#9333ea', tag: 'Endocrinology', Icon: GlucoseDropIcon },
  { title: 'Diabetes (Male)', desc: 'Symptom-based screening: polyuria, polydipsia, and 14 clinical indicators.', href: '/predict/diabetes-male', accent: '#eff6ff', border: '#93c5fd', iconColor: '#2563eb', tag: 'Endocrinology', Icon: SyringeIcon },
  { title: 'Breast Cancer', desc: 'FNA cytology: 30 cell nucleus measurements for malignant vs benign classification.', href: '/predict/breast-cancer', accent: '#f0fdf4', border: '#86efac', iconColor: '#16a34a', tag: 'Oncology', Icon: MicroscopeIcon },
]

function DashboardContent() {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#fafafa' }}>
      <Navbar />
      <main style={{ flex: 1, maxWidth: 1100, margin: '0 auto', width: '100%', padding: '48px 24px 80px' }}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} style={{ marginBottom: 40 }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
            <div>
              <p style={{ fontSize: 12, fontWeight: 700, color: '#0F9D9A', letterSpacing: 2, marginBottom: 6, textTransform: 'uppercase' }}>Clinical Dashboard</p>
              <h1 style={{ fontSize: 32, fontWeight: 800, color: '#09090b', letterSpacing: '-0.8px' }}>AI Diagnostic Suite</h1>
              <p style={{ color: '#71717a', fontSize: 15, marginTop: 6 }}>Select a module to begin your AI-assisted health assessment</p>
            </div>
            <div style={{ background: 'white', border: '1px solid #f0f0f0', borderRadius: 14, padding: '12px 20px', display: 'flex', alignItems: 'center', gap: 10, boxShadow: '0 2px 12px rgba(0,0,0,0.04)' }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#22c55e', animation: 'pulse-dot 2s infinite' }} />
              <span style={{ fontSize: 13, fontWeight: 600, color: '#3f3f46' }}>5 modules active</span>
            </div>
          </div>
        </motion.div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 18, marginBottom: 32 }}>
          {modules.map((m, i) => (
            <motion.div key={m.title} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07 }}>
              <Link href={m.href} style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'white', borderRadius: 20, padding: 28, textDecoration: 'none', border: '1px solid #f0f0f0', boxShadow: '0 2px 12px rgba(0,0,0,0.03)', position: 'relative', overflow: 'hidden', transition: 'all 0.2s' }}
                onMouseEnter={e => { (e.currentTarget.style.transform = 'translateY(-4px)'); (e.currentTarget.style.boxShadow = '0 16px 40px rgba(0,0,0,0.08)'); (e.currentTarget.style.borderColor = m.border) }}
                onMouseLeave={e => { (e.currentTarget.style.transform = ''); (e.currentTarget.style.boxShadow = '0 2px 12px rgba(0,0,0,0.03)'); (e.currentTarget.style.borderColor = '#f0f0f0') }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20 }}>
                  <div style={{ width: 56, height: 56, borderRadius: 16, background: m.accent, border: `1.5px solid ${m.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <m.Icon size={26} color={m.iconColor} strokeWidth={1.7} />
                  </div>
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#71717a', background: '#f4f4f5', padding: '4px 10px', borderRadius: 100, letterSpacing: 0.3 }}>{m.tag}</span>
                </div>
                <h2 style={{ fontWeight: 700, fontSize: 17, color: '#09090b', marginBottom: 8 }}>{m.title}</h2>
                <p style={{ fontSize: 13.5, color: '#71717a', lineHeight: 1.65, flex: 1, marginBottom: 20 }}>{m.desc}</p>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#0F9D9A', fontSize: 13, fontWeight: 700 }}>
                  Start Assessment
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
          style={{ background: 'linear-gradient(135deg, rgba(15,157,154,0.06), rgba(15,157,154,0.03))', border: '1px solid rgba(15,157,154,0.15)', borderRadius: 16, padding: '18px 24px', display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ width: 36, height: 36, borderRadius: 10, background: 'rgba(15,157,154,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0F9D9A" strokeWidth="1.8" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          </div>
          <p style={{ fontSize: 13.5, color: '#134e4a', lineHeight: 1.5 }}>
            <strong>Medical Disclaimer:</strong> SAM AI is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
          </p>
        </motion.div>
      </main>
    </div>
  )
}

export default function DashboardPage() { return <RequireAuth><DashboardContent /></RequireAuth> }
