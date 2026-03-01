'use client'
import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { useAuth } from '@/components/AuthProvider'
import { ActivityIcon, ShieldCheckIcon } from '@/components/MedicalIcons'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

const S = {
  page: { minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24, position: 'relative' as const, overflow: 'hidden', background: 'linear-gradient(135deg, #0a0f1e 0%, #0d1a2e 50%, #0a0f1e 100%)' },
  card: { background: 'rgba(17,24,39,0.85)', backdropFilter: 'blur(24px)', WebkitBackdropFilter: 'blur(24px)', borderRadius: 28, padding: '40px 40px', border: '1px solid rgba(255,255,255,0.10)', boxShadow: '0 32px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(20,184,166,0.08)' },
  label: { display: 'block' as const, fontSize: 13, fontWeight: 600, color: '#94a3b8', marginBottom: 8 },
  input: { width: '100%', padding: '13px 16px', borderRadius: 12, border: '1.5px solid rgba(255,255,255,0.10)', background: 'rgba(255,255,255,0.05)', fontSize: 15, color: '#f1f5f9', transition: 'all 0.15s' },
}

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(''); setLoading(true)
    try {
      const res = await fetch(`${API}/login`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        credentials: 'include', body: JSON.stringify({ email, password }),
      })
      const data = await res.json()
      if (res.ok) {
        login()
        const params = new URLSearchParams(window.location.search)
        router.replace(params.get('returnUrl') || '/dashboard')
      } else setError(data.error || 'Invalid credentials.')
    } catch { setError('Network error. Please try again.') }
    finally { setLoading(false) }
  }

  return (
    <div style={S.page}>
      {/* background blobs */}
      <div className="blob" style={{ width: 600, height: 600, background: 'radial-gradient(circle, rgba(20,184,166,0.12) 0%, transparent 70%)', top: -200, right: -100 }} />
      <div className="blob" style={{ width: 400, height: 400, background: 'radial-gradient(circle, rgba(139,92,246,0.07) 0%, transparent 70%)', bottom: -80, left: -80 }} />

      <motion.div style={{ position: 'relative', zIndex: 10, width: '100%', maxWidth: 440 }}
        initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: 'easeOut' }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <Link href="/" style={{ display: 'inline-flex', alignItems: 'center', gap: 10, textDecoration: 'none', marginBottom: 28 }}>
            <div style={{ width: 40, height: 40, borderRadius: 12, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px rgba(20,184,166,0.35)' }}>
              <ActivityIcon size={18} color="white" strokeWidth={2.5} />
            </div>
            <span style={{ fontWeight: 800, fontSize: 21, color: '#14b8a6' }}>SAM<span style={{ color: '#f1f5f9' }}> AI</span></span>
          </Link>
          <h1 style={{ fontSize: 28, fontWeight: 800, color: '#f1f5f9', marginBottom: 8, letterSpacing: '-0.5px' }}>Welcome back</h1>
          <p style={{ color: '#64748b', fontSize: 15 }}>Sign in to your SAM AI account</p>
        </div>

        {/* Card */}
        <motion.div style={S.card} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {error && (
              <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
                style={{ background: 'rgba(248,113,113,0.10)', border: '1px solid rgba(248,113,113,0.25)', borderRadius: 12, padding: '12px 16px', color: '#f87171', fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                {error}
              </motion.div>
            )}

            {[
              { label: 'Email address', name: 'email', type: 'email', placeholder: 'you@example.com', value: email, setter: setEmail },
              { label: 'Password', name: 'password', type: 'password', placeholder: '••••••••', value: password, setter: setPassword },
            ].map(({ label, name, type, placeholder, value, setter }) => (
              <div key={name}>
                <label style={S.label}>{label}</label>
                <input type={type} name={name} required value={value}
                  onChange={e => setter(e.target.value)} placeholder={placeholder}
                  style={S.input} />
              </div>
            ))}

            <motion.button type="submit" disabled={loading}
              style={{ marginTop: 4, padding: '14px', borderRadius: 12, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 20px rgba(20,184,166,0.30)', letterSpacing: '-0.2px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
              whileHover={loading ? {} : { y: -2, boxShadow: '0 8px 32px rgba(20,184,166,0.40)' }}
              whileTap={{ scale: 0.98 }}>
              {loading ? 'Signing in…' : 'Sign in →'}
            </motion.button>

            <div style={{ textAlign: 'center', paddingTop: 4 }}>
              <span style={{ fontSize: 14, color: '#64748b' }}>No account? </span>
              <Link href="/register" style={{ fontSize: 14, fontWeight: 700, color: '#14b8a6', textDecoration: 'none' }}>Create one free</Link>
            </div>
          </form>
        </motion.div>

        {/* Trust indicator */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7, marginTop: 24 }}>
          <ShieldCheckIcon size={13} color="#64748b" strokeWidth={2} />
          <p style={{ fontSize: 12, color: '#64748b' }}>SAM AI is for educational purposes only. Not a substitute for medical advice.</p>
        </div>
      </motion.div>
    </div>
  )
}
