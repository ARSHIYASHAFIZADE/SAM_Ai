'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import { useAuth } from '@/components/AuthProvider'
import { ActivityIcon, DashboardIcon, LogOutIcon } from '@/components/MedicalIcons'

const NAV_LINKS = [
  { href: '/', label: 'Home' },
  { href: '/about', label: 'About' },
]

export default function Navbar() {
  const { isAuthenticated, logout } = useAuth()
  const [menuOpen, setMenuOpen] = useState(false)
  const pathname = usePathname()

  const isActive = (href: string) =>
    href === '/' ? pathname === '/' : pathname.startsWith(href)

  const linkStyle = (href: string): React.CSSProperties => ({
    textDecoration: 'none',
    fontSize: 14,
    fontWeight: 500,
    color: isActive(href) ? '#14b8a6' : '#94a3b8',
    transition: 'color 0.15s',
    position: 'relative',
  })

  return (
    <header style={{
      position: 'sticky', top: 0, zIndex: 50,
      background: 'rgba(10,15,30,0.85)',
      backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)',
      borderBottom: '1px solid rgba(255,255,255,0.07)',
      boxShadow: '0 1px 24px rgba(0,0,0,0.25)',
    }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 24px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>

        {/* Logo */}
        <Link href="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 10,
            background: 'linear-gradient(135deg,#14b8a6,#0d9488)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 16px rgba(20,184,166,0.35)',
          }}>
            <ActivityIcon size={16} color="white" strokeWidth={2.5} />
          </div>
          <span style={{ fontWeight: 800, fontSize: 19, color: '#14b8a6', letterSpacing: '-0.4px' }}>
            SAM<span style={{ color: '#f1f5f9' }}> AI</span>
          </span>
        </Link>

        {/* Desktop Nav */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: 28 }} className="hidden-mobile">
          {NAV_LINKS.map(({ href, label }) => (
            <Link key={href} href={href} style={linkStyle(href)}
              onMouseEnter={e => { if (!isActive(href)) e.currentTarget.style.color = '#f1f5f9' }}
              onMouseLeave={e => { if (!isActive(href)) e.currentTarget.style.color = '#94a3b8' }}>
              {label}
              {isActive(href) && (
                <span style={{ position: 'absolute', bottom: -20, left: 0, right: 0, height: 2, background: '#14b8a6', borderRadius: 2 }} />
              )}
            </Link>
          ))}
          {isAuthenticated && (
            <Link href="/dashboard" style={linkStyle('/dashboard')}
              onMouseEnter={e => { if (!isActive('/dashboard')) e.currentTarget.style.color = '#f1f5f9' }}
              onMouseLeave={e => { if (!isActive('/dashboard')) e.currentTarget.style.color = '#94a3b8' }}>
              Dashboard
              {isActive('/dashboard') && (
                <span style={{ position: 'absolute', bottom: -20, left: 0, right: 0, height: 2, background: '#14b8a6', borderRadius: 2 }} />
              )}
            </Link>
          )}

          {isAuthenticated ? (
            <button onClick={logout}
              style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '8px 16px', borderRadius: 10, border: '1px solid rgba(248,113,113,0.25)', background: 'rgba(248,113,113,0.08)', color: '#f87171', fontSize: 13, fontWeight: 600, cursor: 'pointer', transition: 'all 0.15s' }}
              onMouseEnter={e => { e.currentTarget.style.background = 'rgba(248,113,113,0.15)'; e.currentTarget.style.borderColor = 'rgba(248,113,113,0.4)' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'rgba(248,113,113,0.08)'; e.currentTarget.style.borderColor = 'rgba(248,113,113,0.25)' }}>
              <LogOutIcon size={13} color="#f87171" strokeWidth={2.2} />
              Sign out
            </button>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Link href="/login"
                style={{ textDecoration: 'none', color: '#94a3b8', fontSize: 14, fontWeight: 500, transition: 'color 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.color = '#f1f5f9'}
                onMouseLeave={e => e.currentTarget.style.color = '#94a3b8'}>
                Sign in
              </Link>
              <Link href="/register"
                style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '8px 18px', borderRadius: 10, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontSize: 13, fontWeight: 700, textDecoration: 'none', boxShadow: '0 2px 14px rgba(20,184,166,0.30)', transition: 'all 0.2s' }}
                onMouseEnter={e => (e.currentTarget.style.transform = 'translateY(-1px)')}
                onMouseLeave={e => (e.currentTarget.style.transform = 'translateY(0)')}>
                Get started
              </Link>
            </div>
          )}
        </nav>

        {/* Mobile burger */}
        <button onClick={() => setMenuOpen(o => !o)} aria-label="Toggle menu"
          style={{ display: 'none', padding: 8, borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)', background: 'transparent', cursor: 'pointer', color: '#94a3b8' }}
          className="show-mobile">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            {menuOpen
              ? <><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></>
              : <><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></>}
          </svg>
        </button>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.07)', background: 'rgba(10,15,30,0.97)', padding: '12px 24px 20px' }}>
          {[...NAV_LINKS, ...(isAuthenticated ? [{ href: '/dashboard', label: 'Dashboard' }] : [])].map(({ href, label }) => (
            <Link key={href} href={href} onClick={() => setMenuOpen(false)}
              style={{ display: 'block', padding: '11px 0', color: isActive(href) ? '#14b8a6' : '#94a3b8', fontWeight: 500, fontSize: 15, textDecoration: 'none', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              {label}
            </Link>
          ))}
          <div style={{ paddingTop: 14 }}>
            {isAuthenticated ? (
              <button onClick={() => { logout(); setMenuOpen(false) }}
                style={{ color: '#f87171', fontWeight: 600, fontSize: 14, background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8 }}>
                <LogOutIcon size={14} color="#f87171" strokeWidth={2.2} /> Sign out
              </button>
            ) : (
              <Link href="/register" onClick={() => setMenuOpen(false)}
                style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '10px 22px', borderRadius: 10, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontWeight: 700, fontSize: 14, textDecoration: 'none' }}>
                Get started
              </Link>
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
