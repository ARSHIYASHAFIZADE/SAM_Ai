'use client'
import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { useAuth } from '@/components/AuthProvider'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
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
      }
      else setError(data.error || 'Invalid credentials.')
    } catch { setError('Network error. Please try again.') }
    finally { setLoading(false) }
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24, position: 'relative', overflow: 'hidden', background: 'linear-gradient(150deg, #f0fafa 0%, #fafafa 50%, #f5f3ff 100%)' }}>
      <div className="blob" style={{ width: 600, height: 600, background: 'radial-gradient(circle, rgba(15,157,154,0.1) 0%, transparent 70%)', top: -150, right: -100 }} />
      <div className="blob" style={{ width: 350, height: 350, background: 'radial-gradient(circle, rgba(139,92,246,0.07) 0%, transparent 70%)', bottom: -80, left: -60 }} />

      <motion.div
        style={{ position: 'relative', zIndex: 10, width: '100%', maxWidth: 440 }}
        initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: 'easeOut' }}
      >
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <Link href="/" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, textDecoration: 'none', marginBottom: 24 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            </div>
            <span style={{ fontWeight: 800, fontSize: 20, color: '#0F9D9A' }}>SAM<span style={{ color: '#09090b' }}> AI</span></span>
          </Link>
          <h1 style={{ fontSize: 28, fontWeight: 800, color: '#09090b', marginBottom: 8, letterSpacing: '-0.5px' }}>Welcome back</h1>
          <p style={{ color: '#71717a', fontSize: 15 }}>Sign in to your SAM AI account</p>
        </div>

        <motion.div
          style={{ background: 'rgba(255,255,255,0.92)', backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)', borderRadius: 24, padding: 36, border: '1px solid rgba(255,255,255,0.9)', boxShadow: '0 20px 60px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.03)' }}
          initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
        >
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
                style={{ background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 12, padding: '12px 16px', color: '#dc2626', fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                {error}
              </motion.div>
            )}

            {[
              { label: 'Email address', name: 'email', type: 'email', placeholder: 'you@example.com', value: email, setter: setEmail },
              { label: 'Password', name: 'password', type: 'password', placeholder: '••••••••', value: password, setter: setPassword },
            ].map(({ label, name, type, placeholder, value, setter }) => (
              <div key={name}>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#3f3f46', marginBottom: 8 }}>{label}</label>
                <input
                  type={type} name={name} required value={value} onChange={e => setter(e.target.value)} placeholder={placeholder}
                  style={{ width: '100%', padding: '12px 16px', borderRadius: 12, border: '1.5px solid #e4e4e7', background: '#fafafa', fontSize: 15, color: '#09090b', transition: 'all 0.15s' }}
                />
              </div>
            ))}

            <motion.button
              type="submit" disabled={loading}
              style={{ padding: '14px', borderRadius: 12, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 16px rgba(15,157,154,0.3)', letterSpacing: '-0.2px' }}
              whileHover={loading ? {} : { y: -2, boxShadow: '0 8px 28px rgba(15,157,154,0.4)' }}
              whileTap={{ scale: 0.98 }}
            >
              {loading ? 'Signing in…' : 'Sign in →'}
            </motion.button>

            <div style={{ textAlign: 'center', paddingTop: 4 }}>
              <span style={{ fontSize: 14, color: '#71717a' }}>No account? </span>
              <Link href="/register" style={{ fontSize: 14, fontWeight: 700, color: '#0F9D9A', textDecoration: 'none' }}>Create one free</Link>
            </div>
          </form>
        </motion.div>

        <p style={{ textAlign: 'center', marginTop: 20, fontSize: 12, color: '#a1a1aa', lineHeight: 1.5 }}>
          SAM AI is for educational purposes only.<br />Not a substitute for medical advice.
        </p>
      </motion.div>
    </div>
  )
}
