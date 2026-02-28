'use client'
import { motion } from 'framer-motion'
import Link from 'next/link'

export default function AboutPage() {
  const cards = [
    { title: 'Mission', icon: '🎯', text: 'Make AI health screening accessible to everyone. Early detection saves lives.' },
    { title: 'Technology', icon: '⚙️', text: 'Gradient Boosting, Logistic Regression, and ensemble ML models trained on validated clinical datasets.' },
    { title: 'Privacy', icon: '🔒', text: 'No health data is stored. All predictions are stateless and session-scoped.' },
    { title: 'Disclaimer', icon: '⚕️', text: 'For educational use only. Always consult a qualified healthcare professional.' },
  ]

  return (
    <div style={{ minHeight: '100vh', background: '#fafafa' }}>
      <header style={{ background: 'rgba(255,255,255,0.92)', backdropFilter: 'blur(16px)', borderBottom: '1px solid rgba(0,0,0,0.06)', padding: '0 24px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'sticky', top: 0, zIndex: 50, boxShadow: '0 1px 20px rgba(0,0,0,0.05)' }}>
        <Link href="/" style={{ display: 'flex', alignItems: 'center', gap: 8, textDecoration: 'none' }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
          </div>
          <span style={{ fontWeight: 800, fontSize: 18, color: '#0F9D9A' }}>SAM<span style={{ color: '#09090b' }}> AI</span></span>
        </Link>
        <Link href="/" style={{ fontSize: 14, color: '#71717a', textDecoration: 'none' }}>← Back to home</Link>
      </header>

      <main style={{ maxWidth: 900, margin: '0 auto', padding: '72px 24px 160px' }}>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <div style={{ display: 'inline-block', padding: '5px 14px', borderRadius: 100, background: 'rgba(15,157,154,0.07)', border: '1px solid rgba(15,157,154,0.2)', fontSize: 12, fontWeight: 700, color: '#0F9D9A', marginBottom: 20, letterSpacing: 0.5 }}>About SAM AI</div>
          <h1 style={{ fontSize: 'clamp(2rem, 4vw, 3rem)', fontWeight: 800, color: '#09090b', letterSpacing: '-1px', marginBottom: 20, lineHeight: 1.15 }}>
            AI-powered diagnostics<br />
            <span style={{ background: 'linear-gradient(135deg,#0F9D9A,#0a6f6d)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
              built for early detection
            </span>
          </h1>
          <p style={{ fontSize: 17, color: '#52525b', lineHeight: 1.75, maxWidth: 620, marginBottom: 56 }}>
            SAM AI (Smart Adaptive Medical AI) is an educational platform that leverages machine learning to assist in the early detection of critical health conditions including heart disease, diabetes, liver disease, and cancer.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 20, marginBottom: 56 }}>
            {cards.map((c, i) => (
              <motion.div
                key={c.title}
                style={{ background: 'white', borderRadius: 20, padding: 28, border: '1px solid #f0f0f0', boxShadow: '0 2px 12px rgba(0,0,0,0.04)' }}
                initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 + i * 0.07 }}
              >
                <div style={{ fontSize: 28, marginBottom: 14 }}>{c.icon}</div>
                <h3 style={{ fontWeight: 700, fontSize: 16, color: '#0F9D9A', marginBottom: 10 }}>{c.title}</h3>
                <p style={{ fontSize: 14, color: '#71717a', lineHeight: 1.65 }}>{c.text}</p>
              </motion.div>
            ))}
          </div>

          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            <Link href="/register" style={{ display: 'inline-block', padding: '14px 28px', borderRadius: 12, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', color: 'white', fontWeight: 700, fontSize: 15, textDecoration: 'none', boxShadow: '0 4px 20px rgba(15,157,154,0.3)' }}>
              Get started free →
            </Link>
            <Link href="/" style={{ display: 'inline-block', padding: '14px 28px', borderRadius: 12, background: 'white', color: '#3f3f46', fontWeight: 600, fontSize: 15, textDecoration: 'none', border: '1px solid #e4e4e7' }}>
              Back to home
            </Link>
          </div>
        </motion.div>
      </main>
    </div>
  )
}
