'use client'
import Link from 'next/link'
import { motion } from 'framer-motion'

export default function NotFound() {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fafafa', padding: 24 }}>
      <div className="blob" style={{ width: 500, height: 500, background: 'radial-gradient(circle, rgba(15,157,154,0.1) 0%, transparent 70%)', top: -100, right: -100, position: 'fixed' }} />
      <motion.div
        initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
        style={{ textAlign: 'center', maxWidth: 480 }}
      >
        <div style={{ fontSize: 72, marginBottom: 16, lineHeight: 1 }}>404</div>
        <h1 style={{ fontSize: 28, fontWeight: 800, color: '#09090b', marginBottom: 12, letterSpacing: '-0.5px' }}>Page not found</h1>
        <p style={{ color: '#71717a', fontSize: 16, lineHeight: 1.7, marginBottom: 36 }}>
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
        </p>
        <motion.div whileHover={{ y: -2 }}>
          <Link href="/" style={{ display: 'inline-block', padding: '14px 32px', borderRadius: 12, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', color: 'white', fontWeight: 700, fontSize: 15, textDecoration: 'none', boxShadow: '0 4px 20px rgba(15,157,154,0.3)' }}>
            ← Back to home
          </Link>
        </motion.div>
      </motion.div>
    </div>
  )
}
