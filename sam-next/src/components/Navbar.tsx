'use client'
import Link from 'next/link'
import { useState } from 'react'
import { useAuth } from '@/components/AuthProvider'

export default function Navbar() {
  const { isAuthenticated, logout } = useAuth()
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <header style={{
      position: 'sticky', top: 0, zIndex: 50,
      background: 'rgba(255,255,255,0.92)',
      backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)',
      borderBottom: '1px solid rgba(0,0,0,0.06)',
      boxShadow: '0 1px 20px rgba(0,0,0,0.05)',
    }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 24px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Link href="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
          </div>
          <span style={{ fontWeight: 800, fontSize: 20, color: '#0F9D9A', letterSpacing: '-0.5px' }}>SAM<span style={{ color: '#18181b' }}> AI</span></span>
        </Link>

        <nav style={{ display: 'flex', alignItems: 'center', gap: 32 }} className="hidden-mobile">
          {[['/', 'Home'], ['/about', 'About']].map(([href, label]) => (
            <Link key={href} href={href} style={{ textDecoration: 'none', color: '#52525b', fontSize: 14, fontWeight: 500, transition: 'color 0.15s' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#0F9D9A')}
              onMouseLeave={e => (e.currentTarget.style.color = '#52525b')}
            >{label}</Link>
          ))}
          {isAuthenticated && (
            <Link href="/dashboard" style={{ textDecoration: 'none', color: '#52525b', fontSize: 14, fontWeight: 500, transition: 'color 0.15s' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#0F9D9A')}
              onMouseLeave={e => (e.currentTarget.style.color = '#52525b')}
            >Dashboard</Link>
          )}
          {isAuthenticated ? (
            <button onClick={logout} style={{ padding: '8px 18px', borderRadius: 10, border: '1px solid #e4e4e7', background: 'white', color: '#dc2626', fontSize: 13, fontWeight: 600, cursor: 'pointer', transition: 'all 0.15s' }}
              onMouseEnter={e => { (e.currentTarget.style.background = '#fef2f2'); (e.currentTarget.style.borderColor = '#fca5a5'); }}
              onMouseLeave={e => { (e.currentTarget.style.background = 'white'); (e.currentTarget.style.borderColor = '#e4e4e7'); }}>
              Sign out
            </button>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <Link href="/login" style={{ textDecoration: 'none', color: '#52525b', fontSize: 14, fontWeight: 500 }}
                onMouseEnter={e => (e.currentTarget.style.color = '#0F9D9A')}
                onMouseLeave={e => (e.currentTarget.style.color = '#52525b')}
              >Sign in</Link>
              <Link href="/register" style={{ padding: '8px 18px', borderRadius: 10, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', color: 'white', fontSize: 13, fontWeight: 600, textDecoration: 'none', boxShadow: '0 2px 12px rgba(15,157,154,0.3)', transition: 'all 0.2s' }}
                onMouseEnter={e => (e.currentTarget.style.transform = 'translateY(-1px)')}
                onMouseLeave={e => (e.currentTarget.style.transform = 'translateY(0)')}
              >Get started</Link>
            </div>
          )}
        </nav>

        <button onClick={() => setMenuOpen(o => !o)} aria-label="Toggle menu"
          style={{ display: 'none', padding: 8, borderRadius: 8, border: 'none', background: 'transparent', cursor: 'pointer', color: '#52525b' }}
          className="show-mobile">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            {menuOpen ? <><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></> : <><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></>}
          </svg>
        </button>
      </div>

      {menuOpen && (
        <div style={{ borderTop: '1px solid #f4f4f5', background: 'white', padding: '12px 24px 16px' }}>
          {[['/', 'Home'], ['/about', 'About'], ...(isAuthenticated ? [['/dashboard', 'Dashboard']] : [])].map(([href, label]) => (
            <Link key={href} href={href} onClick={() => setMenuOpen(false)}
              style={{ display: 'block', padding: '10px 0', color: '#3f3f46', fontWeight: 500, fontSize: 14, textDecoration: 'none', borderBottom: '1px solid #f4f4f5' }}>{label}</Link>
          ))}
          <div style={{ paddingTop: 12 }}>
            {isAuthenticated ? (
              <button onClick={() => { logout(); setMenuOpen(false) }} style={{ color: '#dc2626', fontWeight: 600, fontSize: 14, background: 'none', border: 'none', cursor: 'pointer' }}>Sign out</button>
            ) : (
              <Link href="/register" onClick={() => setMenuOpen(false)} style={{ display: 'inline-block', padding: '10px 20px', borderRadius: 10, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', color: 'white', fontWeight: 600, fontSize: 14, textDecoration: 'none' }}>Get started</Link>
            )}
          </div>
        </div>
      )}

      <style>{`
        @media (max-width: 768px) {
          .hidden-mobile { display: none !important; }
          .show-mobile { display: flex !important; }
        }
      `}</style>
    </header>
  )
}
