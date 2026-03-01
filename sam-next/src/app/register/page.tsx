'use client'
import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { ActivityIcon, ShieldCheckIcon } from '@/components/MedicalIcons'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

const S = {
  page: { minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24, position: 'relative' as const, overflow: 'hidden', background: 'linear-gradient(135deg, #0a0f1e 0%, #0f0d1e 50%, #0a0f1e 100%)' },
  card: { background: 'rgba(17,24,39,0.85)', backdropFilter: 'blur(24px)', WebkitBackdropFilter: 'blur(24px)', borderRadius: 28, padding: '40px 40px', border: '1px solid rgba(255,255,255,0.10)', boxShadow: '0 32px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(20,184,166,0.08)' },
  label: { display: 'block' as const, fontSize: 13, fontWeight: 600, color: '#94a3b8', marginBottom: 8 },
  input: { width: '100%', padding: '13px 16px', borderRadius: 12, border: '1.5px solid rgba(255,255,255,0.10)', background: 'rgba(255,255,255,0.05)', fontSize: 15, color: '#f1f5f9', transition: 'all 0.15s' },
}

export default function RegisterPage() {
  const [form, setForm] = useState({ name: '', email: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(''); setLoading(true)
    try {
      const res = await fetch(`${API}/register`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        credentials: 'include', body: JSON.stringify(form),
      })
      const data = await res.json()
      if (res.ok) router.replace('/login')
      else setError(data.error || 'Registration failed.')
    } catch { setError('Network error. Please try again.') }
    finally { setLoading(false) }
  }

  const fields = [
    { key: 'name' as const, label: 'Full name', type: 'text', placeholder: 'Jane Doe' },
    { key: 'email' as const, label: 'Email address', type: 'email', placeholder: 'you@example.com' },
    { key: 'password' as const, label: 'Password', type: 'password', placeholder: 'At least 8 characters' },
  ]

  return (
    <div style={S.page}>
      <div className="blob" style={{ width: 500, height: 500, background: 'radial-gradient(circle, rgba(20,184,166,0.10) 0%, transparent 70%)', top: -120, left: -80 }} />
      <div className="blob" style={{ width: 400, height: 400, background: 'radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%)', bottom: -80, right: -60 }} />

      <motion.div style={{ position: 'relative', zIndex: 10, width: '100%', maxWidth: 440 }}
        initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: 'easeOut' }}>

        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <Link href="/" style={{ display: 'inline-flex', alignItems: 'center', gap: 10, textDecoration: 'none', marginBottom: 28 }}>
            <div style={{ width: 40, height: 40, borderRadius: 12, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px rgba(20,184,166,0.35)' }}>
              <ActivityIcon size={18} color="white" strokeWidth={2.5} />
            </div>
            <span style={{ fontWeight: 800, fontSize: 21, color: '#14b8a6' }}>SAM<span style={{ color: '#f1f5f9' }}> AI</span></span>
          </Link>
          <h1 style={{ fontSize: 28, fontWeight: 800, color: '#f1f5f9', marginBottom: 8, letterSpacing: '-0.5px' }}>Create your account</h1>
          <p style={{ color: '#64748b', fontSize: 15 }}>Start your health journey today — it&apos;s free</p>
        </div>

        <motion.div style={S.card} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {error && (
              <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
                style={{ background: 'rgba(248,113,113,0.10)', border: '1px solid rgba(248,113,113,0.25)', borderRadius: 12, padding: '12px 16px', color: '#f87171', fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                {error}
              </motion.div>
            )}

            {fields.map(({ key, label, type, placeholder }) => (
              <div key={key}>
                <label style={S.label}>{label}</label>
                <input type={type} required value={form[key]}
                  onChange={e => setForm({ ...form, [key]: e.target.value })} placeholder={placeholder}
                  style={S.input} />
              </div>
            ))}

            <motion.button type="submit" disabled={loading}
              style={{ marginTop: 4, padding: '14px', borderRadius: 12, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 20px rgba(20,184,166,0.30)', letterSpacing: '-0.2px' }}
              whileHover={loading ? {} : { y: -2, boxShadow: '0 8px 32px rgba(20,184,166,0.40)' }}
              whileTap={{ scale: 0.98 }}>
              {loading ? 'Creating account…' : 'Create account →'}
            </motion.button>

            <div style={{ textAlign: 'center', paddingTop: 4 }}>
              <span style={{ fontSize: 14, color: '#64748b' }}>Already have an account? </span>
              <Link href="/login" style={{ fontSize: 14, fontWeight: 700, color: '#14b8a6', textDecoration: 'none' }}>Sign in</Link>
            </div>
          </form>
        </motion.div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7, marginTop: 24 }}>
          <ShieldCheckIcon size={13} color="#64748b" strokeWidth={2} />
          <p style={{ fontSize: 12, color: '#64748b' }}>SAM AI is for educational purposes only. Not a substitute for medical advice.</p>
        </div>
      </motion.div>
    </div>
  )
}
