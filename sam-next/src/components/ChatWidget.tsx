'use client'
import { useState, useRef, useEffect } from 'react'
import { SparklesIcon } from '@/components/MedicalIcons'

interface Message { role: 'user' | 'assistant'; text: string }

const TEAL = '#0F9D9A'

export default function ChatWidget() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', text: 'Hi! I\'m SAM, your medical AI assistant. Ask me anything about health conditions, risk factors, or how to use the diagnostic modules.' },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])
  useEffect(() => { if (open) setTimeout(() => inputRef.current?.focus(), 200) }, [open])

  const send = async () => {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    setMessages(m => [...m, { role: 'user', text }])
    setLoading(true)
    try {
      const res = await fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
      const data = await res.json()
      setMessages(m => [...m, { role: 'assistant', text: data.reply || data.error || 'No response generated.' }])
    } catch {
      setMessages(m => [...m, { role: 'assistant', text: 'Connection error. Please try again.' }])
    } finally { setLoading(false) }
  }

  return (
    <>
      {/* ── CHAT PANEL ── */}
      {open && (
        <div style={{
          position: 'fixed', bottom: 88, right: 24, zIndex: 50,
          width: 380, maxHeight: 540,
          borderRadius: 24, overflow: 'hidden',
          background: 'white',
          boxShadow: '0 24px 80px rgba(0,0,0,0.13), 0 0 0 1px rgba(0,0,0,0.05)',
          display: 'flex', flexDirection: 'column',
          animation: 'chatOpen 0.22s cubic-bezier(0.34,1.56,0.64,1)',
        }}>
          {/* Header */}
          <div style={{ padding: '16px 20px', background: `linear-gradient(135deg, ${TEAL}, #0d7d7a)`, display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 40, height: 40, borderRadius: 12, background: 'rgba(255,255,255,0.18)', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid rgba(255,255,255,0.25)' }}>
              <SparklesIcon size={18} color="white" strokeWidth={1.8} />
            </div>
            <div style={{ flex: 1 }}>
              <p style={{ color: 'white', fontWeight: 700, fontSize: 15, lineHeight: 1 }}>SAM Assistant</p>
              <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: 12, marginTop: 3 }}>Clinical AI Support</p>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(255,255,255,0.15)', padding: '4px 10px', borderRadius: 100 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#4ade80' }} />
              <span style={{ color: 'rgba(255,255,255,0.9)', fontSize: 11, fontWeight: 600 }}>Online</span>
            </div>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: 12, minHeight: 0 }}>
            {messages.map((m, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', gap: 8, alignItems: 'flex-end' }}>
                {m.role === 'assistant' && (
                  <div style={{ width: 30, height: 30, borderRadius: 9, background: `rgba(15,157,154,0.1)`, border: '1px solid rgba(15,157,154,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginBottom: 2 }}>
                    <SparklesIcon size={13} color={TEAL} strokeWidth={1.8} />
                  </div>
                )}
                <div style={{
                  maxWidth: '76%', padding: '10px 14px',
                  borderRadius: m.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                  background: m.role === 'user' ? `linear-gradient(135deg, ${TEAL}, #0d7d7a)` : '#f4f4f5',
                  color: m.role === 'user' ? 'white' : '#09090b',
                  fontSize: 14, lineHeight: 1.55,
                }}>
                  {m.text}
                </div>
              </div>
            ))}
            {loading && (
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8 }}>
                <div style={{ width: 30, height: 30, borderRadius: 9, background: 'rgba(15,157,154,0.1)', border: '1px solid rgba(15,157,154,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <SparklesIcon size={13} color={TEAL} strokeWidth={1.8} />
                </div>
                <div style={{ background: '#f4f4f5', padding: '12px 16px', borderRadius: '16px 16px 16px 4px', display: 'flex', gap: 5, alignItems: 'center' }}>
                  {[0, 1, 2].map(n => (
                    <span key={n} style={{ width: 6, height: 6, borderRadius: '50%', background: TEAL, display: 'block', animation: `bounce 1.2s ${n * 0.18}s infinite ease-in-out` }} />
                  ))}
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>

          {/* Input */}
          <div style={{ padding: '12px 14px', borderTop: '1px solid #f4f4f5' }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input ref={inputRef} value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && send()} disabled={loading}
                placeholder="Ask a health question…"
                style={{ flex: 1, padding: '10px 14px', borderRadius: 12, border: '1.5px solid #e4e4e7', background: '#fafafa', fontSize: 14, color: '#09090b', outline: 'none', transition: 'all 0.15s' }}
                onFocus={e => { e.target.style.borderColor = TEAL; e.target.style.background = 'white'; e.target.style.boxShadow = `0 0 0 3px rgba(15,157,154,0.12)` }}
                onBlur={e => { e.target.style.borderColor = '#e4e4e7'; e.target.style.background = '#fafafa'; e.target.style.boxShadow = 'none' }} />
              <button onClick={send} disabled={loading || !input.trim()}
                style={{ width: 40, height: 40, borderRadius: 12, background: input.trim() && !loading ? `linear-gradient(135deg, ${TEAL}, #0d7d7a)` : '#f4f4f5', border: 'none', cursor: input.trim() && !loading ? 'pointer' : 'not-allowed', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.15s', flexShrink: 0 }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={input.trim() && !loading ? 'white' : '#a1a1aa'} strokeWidth="2.2" strokeLinecap="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
              </button>
            </div>
            <p style={{ fontSize: 11, color: '#a1a1aa', textAlign: 'center', marginTop: 8 }}>Not a substitute for medical advice</p>
          </div>
        </div>
      )}

      {/* ── FAB: AI CARD BUTTON ── */}
      <button onClick={() => setOpen(o => !o)} aria-label="Open AI assistant"
        style={{
          position: 'fixed', bottom: 24, right: 24, zIndex: 50,
          height: 52,
          paddingLeft: open ? 18 : 14,
          paddingRight: 20,
          borderRadius: 16,
          background: open
            ? 'white'
            : `linear-gradient(135deg, ${TEAL}, #0d7d7a)`,
          border: open ? '1.5px solid #e4e4e7' : 'none',
          cursor: 'pointer',
          display: 'flex', alignItems: 'center', gap: 10,
          boxShadow: open
            ? '0 4px 20px rgba(0,0,0,0.1)'
            : '0 4px 24px rgba(15,157,154,0.38), 0 1px 6px rgba(15,157,154,0.2)',
          transition: 'all 0.25s cubic-bezier(0.34,1.56,0.64,1)',
        }}
        onMouseEnter={e => (e.currentTarget.style.transform = 'translateY(-2px)')}
        onMouseLeave={e => (e.currentTarget.style.transform = '')}
      >
        {open ? (
          <>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#71717a" strokeWidth="2.2" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            <span style={{ fontSize: 14, fontWeight: 600, color: '#52525b', letterSpacing: '-0.2px' }}>Close</span>
          </>
        ) : (
          <>
            <div style={{ width: 26, height: 26, borderRadius: 8, background: 'rgba(255,255,255,0.22)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <SparklesIcon size={14} color="white" strokeWidth={2} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: 'white', lineHeight: 1.1, letterSpacing: '-0.2px' }}>Ask SAM AI</span>
              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.72)', fontWeight: 500, letterSpacing: 0.2 }}>Medical Assistant</span>
            </div>
          </>
        )}
      </button>

      <style>{`
        @keyframes bounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-5px); } }
        @keyframes chatOpen { from { opacity: 0; transform: translateY(14px) scale(0.96); } to { opacity: 1; transform: translateY(0) scale(1); } }
      `}</style>
    </>
  )
}
