'use client'
import { useState, useRef, useEffect, useCallback } from 'react'
import { SparklesIcon } from '@/components/MedicalIcons'

interface Message { role: 'user' | 'assistant'; text: string; ts: number }
interface ChatSession { id: string; title: string; date: string; messages: Message[] }

const STORAGE_KEY = 'sam_chat_sessions'
const MAX_SESSIONS = 15

const WELCOME: Message = {
  role: 'assistant',
  text: "Hi, I'm SAM — your clinical AI assistant. Ask me anything about health conditions, risk factors, or how to interpret your diagnostic results.",
  ts: 0,
}

const QUICK_PROMPTS = [
  'What is normal cholesterol?',
  'How is diabetes diagnosed?',
  'What does high BMI mean?',
  'Explain bilirubin and liver enzymes',
]

function loadSessions(): ChatSession[] {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') } catch { return [] }
}
function saveSessions(sessions: ChatSession[]) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(0, MAX_SESSIONS))) } catch { /* ignore */ }
}
function fmtTime(ts: number) {
  if (!ts) return ''
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}
function fmtDate(iso: string) {
  return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export default function ChatWidget() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([{ ...WELCOME, ts: Date.now() }])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [isAtBottom, setIsAtBottom] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { setSessions(loadSessions()) }, [])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])
  useEffect(() => { if (open && !showHistory) setTimeout(() => inputRef.current?.focus(), 200) }, [open, showHistory])
  useEffect(() => {
    const onScroll = () => setIsAtBottom(window.innerHeight + window.scrollY >= document.body.offsetHeight - 60)
    window.addEventListener('scroll', onScroll)
    onScroll()
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const saveCurrentSession = useCallback((msgs: Message[], sessionId: string) => {
    if (msgs.filter(m => m.role === 'user').length === 0) return
    const firstUserMsg = msgs.find(m => m.role === 'user')?.text ?? 'Conversation'
    const title = firstUserMsg.length > 40 ? firstUserMsg.slice(0, 40) + '…' : firstUserMsg
    setSessions(prev => {
      const existing = prev.find(s => s.id === sessionId)
      const updated: ChatSession[] = existing
        ? prev.map(s => s.id === sessionId ? { ...s, messages: msgs } : s)
        : [{ id: sessionId, title, date: new Date().toISOString(), messages: msgs }, ...prev]
      saveSessions(updated)
      return updated
    })
  }, [])

  const send = async (quickText?: string) => {
    const msgText = (quickText ?? input).trim()
    if (!msgText || loading) return
    setInput('')

    let sessionId = currentSessionId
    if (!sessionId) {
      sessionId = Date.now().toString()
      setCurrentSessionId(sessionId)
    }

    const newMessages: Message[] = [...messages, { role: 'user', text: msgText, ts: Date.now() }]
    setMessages(newMessages)
    setLoading(true)
    try {
      const res = await fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msgText }),
      })
      const data = await res.json()
      const reply = data.reply || data.error || 'No response generated.'
      const finalMessages: Message[] = [...newMessages, { role: 'assistant', text: reply, ts: Date.now() }]
      setMessages(finalMessages)
      saveCurrentSession(finalMessages, sessionId)
    } catch {
      setMessages([...newMessages, { role: 'assistant', text: 'Connection error. Please try again.', ts: Date.now() }])
    } finally { setLoading(false) }
  }

  const newSession = () => {
    if (currentSessionId) saveCurrentSession(messages, currentSessionId)
    setMessages([{ ...WELCOME, ts: Date.now() }])
    setCurrentSessionId(null)
    setShowHistory(false)
  }

  const loadSession = (session: ChatSession) => {
    setMessages(session.messages)
    setCurrentSessionId(session.id)
    setShowHistory(false)
  }

  const deleteSession = (id: string) => {
    setSessions(prev => {
      const updated = prev.filter(s => s.id !== id)
      saveSessions(updated)
      return updated
    })
  }

  const userMessageCount = messages.filter(m => m.role === 'user').length
  const bottom = (isAtBottom ? 50 : 0) + 24

  return (
    <>
      {/* ── CHAT PANEL ── */}
      {open && (
        <div style={{
          position: 'fixed', bottom: bottom + 62, right: 24, zIndex: 9990,
          width: 384, height: 560,
          borderRadius: 22, overflow: 'hidden',
          background: '#111827',
          border: '1px solid rgba(255,255,255,0.08)',
          boxShadow: '0 24px 80px rgba(0,0,0,0.65), 0 0 0 1px rgba(20,184,166,0.08)',
          display: 'flex', flexDirection: 'column',
          animation: 'chatOpen 0.22s cubic-bezier(0.34,1.56,0.64,1)',
        }}>

          {/* ── Header ── */}
          <div style={{ padding: '13px 14px', background: 'rgba(10,15,30,0.97)', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', gap: 9, flexShrink: 0 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: 'rgba(20,184,166,0.12)', border: '1.5px solid rgba(20,184,166,0.28)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 14px rgba(20,184,166,0.14)', flexShrink: 0 }}>
              <SparklesIcon size={16} color="#14b8a6" strokeWidth={1.9} />
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <p style={{ color: '#f1f5f9', fontWeight: 700, fontSize: 13.5, lineHeight: 1 }}>SAM Assistant</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 3 }}>
                <div style={{ width: 5, height: 5, borderRadius: '50%', background: '#4ade80' }} />
                <p style={{ color: '#475569', fontSize: 11 }}>Clinical AI Support</p>
              </div>
            </div>

            {/* New session */}
            <button onClick={newSession} title="New conversation"
              style={{ width: 28, height: 28, borderRadius: 7, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', transition: 'all 0.15s' }}
              onMouseEnter={e => { e.currentTarget.style.background = 'rgba(20,184,166,0.10)'; e.currentTarget.style.color = '#14b8a6'; e.currentTarget.style.borderColor = 'rgba(20,184,166,0.25)' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; e.currentTarget.style.color = '#475569'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)' }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
              </svg>
            </button>

            {/* History */}
            <button onClick={() => setShowHistory(h => !h)} title="Chat history"
              style={{ width: 28, height: 28, borderRadius: 7, background: showHistory ? 'rgba(20,184,166,0.12)' : 'rgba(255,255,255,0.04)', border: showHistory ? '1px solid rgba(20,184,166,0.28)' : '1px solid rgba(255,255,255,0.07)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: showHistory ? '#14b8a6' : '#475569', transition: 'all 0.15s' }}
              onMouseEnter={e => { if (!showHistory) { e.currentTarget.style.background = 'rgba(20,184,166,0.08)'; e.currentTarget.style.color = '#14b8a6'; e.currentTarget.style.borderColor = 'rgba(20,184,166,0.20)' } }}
              onMouseLeave={e => { if (!showHistory) { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; e.currentTarget.style.color = '#475569'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)' } }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
              </svg>
            </button>

            {/* Close */}
            <button onClick={() => setOpen(false)}
              style={{ width: 28, height: 28, borderRadius: 7, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', transition: 'all 0.15s' }}
              onMouseEnter={e => { e.currentTarget.style.background = 'rgba(248,113,113,0.10)'; e.currentTarget.style.color = '#f87171'; e.currentTarget.style.borderColor = 'rgba(248,113,113,0.25)' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; e.currentTarget.style.color = '#475569'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)' }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
            </button>
          </div>

          {/* ── History Panel ── */}
          {showHistory ? (
            <div style={{ flex: 1, overflowY: 'auto', padding: '14px 12px', display: 'flex', flexDirection: 'column', gap: 6 }}>
              <p style={{ fontSize: 10, fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: 1.2, marginBottom: 6 }}>Past Conversations</p>
              {sessions.length === 0 ? (
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10, paddingTop: 60 }}>
                  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="1.5" strokeLinecap="round"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
                  <p style={{ fontSize: 13, color: '#475569', textAlign: 'center' }}>No saved conversations yet.</p>
                </div>
              ) : sessions.map(s => (
                <div key={s.id} onClick={() => loadSession(s)}
                  style={{ padding: '11px 13px', borderRadius: 11, background: s.id === currentSessionId ? 'rgba(20,184,166,0.08)' : 'rgba(255,255,255,0.03)', border: s.id === currentSessionId ? '1px solid rgba(20,184,166,0.20)' : '1px solid rgba(255,255,255,0.06)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 10, transition: 'all 0.15s' }}
                  onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.background = 'rgba(20,184,166,0.06)'; (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(20,184,166,0.18)' }}
                  onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.background = s.id === currentSessionId ? 'rgba(20,184,166,0.08)' : 'rgba(255,255,255,0.03)'; (e.currentTarget as HTMLDivElement).style.borderColor = s.id === currentSessionId ? 'rgba(20,184,166,0.20)' : 'rgba(255,255,255,0.06)' }}
                >
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ fontSize: 13, color: '#e2e8f0', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.title}</p>
                    <p style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>{fmtDate(s.date)} · {s.messages.filter(m => m.role === 'user').length} messages</p>
                  </div>
                  <button
                    onClick={e => { e.stopPropagation(); deleteSession(s.id) }}
                    style={{ width: 24, height: 24, borderRadius: 6, background: 'transparent', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', flexShrink: 0, transition: 'all 0.15s' }}
                    onMouseEnter={e => { e.currentTarget.style.background = 'rgba(248,113,113,0.10)'; e.currentTarget.style.color = '#f87171' }}
                    onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#475569' }}
                  >
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14H6L5 6" /><path d="M10 11v6m4-6v6" /><path d="M9 6V4h6v2" /></svg>
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <>
              {/* ── Messages ── */}
              <div style={{ flex: 1, overflowY: 'auto', padding: '14px 12px 8px', display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0 }}>
                {messages.map((m, i) => (
                  <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
                    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8 }}>
                      {m.role === 'assistant' && (
                        <div style={{ width: 26, height: 26, borderRadius: 7, background: 'rgba(20,184,166,0.10)', border: '1px solid rgba(20,184,166,0.18)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginBottom: 2 }}>
                          <SparklesIcon size={11} color="#14b8a6" strokeWidth={1.9} />
                        </div>
                      )}
                      <div style={{
                        maxWidth: '78%', padding: '9px 12px',
                        borderRadius: m.role === 'user' ? '15px 15px 4px 15px' : '15px 15px 15px 4px',
                        background: m.role === 'user' ? 'rgba(20,184,166,0.12)' : '#1a2236',
                        border: m.role === 'user' ? '1px solid rgba(20,184,166,0.22)' : '1px solid rgba(255,255,255,0.06)',
                        color: '#e2e8f0', fontSize: 13.5, lineHeight: 1.6,
                      }}>
                        {m.text}
                      </div>
                    </div>
                    {m.ts > 0 && (
                      <p style={{ fontSize: 10, color: '#334155', marginTop: 3, marginLeft: m.role === 'assistant' ? 34 : 0 }}>{fmtTime(m.ts)}</p>
                    )}
                  </div>
                ))}

                {/* Quick prompts — only when no user messages sent */}
                {userMessageCount === 0 && (
                  <div style={{ marginTop: 8 }}>
                    <p style={{ fontSize: 10, color: '#475569', marginBottom: 8, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase' }}>Quick questions</p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {QUICK_PROMPTS.map(q => (
                        <button key={q} onClick={() => send(q)}
                          style={{ padding: '6px 12px', borderRadius: 20, background: 'rgba(20,184,166,0.06)', border: '1px solid rgba(20,184,166,0.18)', color: '#94a3b8', fontSize: 12, cursor: 'pointer', transition: 'all 0.15s', textAlign: 'left', lineHeight: 1.4 }}
                          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(20,184,166,0.12)'; e.currentTarget.style.color = '#14b8a6'; e.currentTarget.style.borderColor = 'rgba(20,184,166,0.35)' }}
                          onMouseLeave={e => { e.currentTarget.style.background = 'rgba(20,184,166,0.06)'; e.currentTarget.style.color = '#94a3b8'; e.currentTarget.style.borderColor = 'rgba(20,184,166,0.18)' }}
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Loading dots */}
                {loading && (
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8 }}>
                    <div style={{ width: 26, height: 26, borderRadius: 7, background: 'rgba(20,184,166,0.10)', border: '1px solid rgba(20,184,166,0.18)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <SparklesIcon size={11} color="#14b8a6" strokeWidth={1.9} />
                    </div>
                    <div style={{ background: '#1a2236', border: '1px solid rgba(255,255,255,0.06)', padding: '11px 15px', borderRadius: '15px 15px 15px 4px', display: 'flex', gap: 4, alignItems: 'center' }}>
                      {[0, 1, 2].map(n => (
                        <span key={n} style={{ width: 5, height: 5, borderRadius: '50%', background: '#14b8a6', display: 'block', animation: `bounce 1.2s ${n * 0.18}s infinite ease-in-out`, opacity: 0.7 }} />
                      ))}
                    </div>
                  </div>
                )}
                <div ref={endRef} />
              </div>

              {/* ── Input ── */}
              <div style={{ padding: '10px 12px 12px', borderTop: '1px solid rgba(255,255,255,0.05)', flexShrink: 0 }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    ref={inputRef} value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
                    disabled={loading}
                    placeholder="Ask a health question…"
                    style={{ flex: 1, padding: '9px 13px', borderRadius: 11, border: '1.5px solid rgba(255,255,255,0.09)', background: 'rgba(255,255,255,0.04)', fontSize: 13.5, color: '#f1f5f9', outline: 'none', transition: 'all 0.15s' }}
                    onFocus={e => { e.target.style.borderColor = '#14b8a6'; e.target.style.boxShadow = '0 0 0 3px rgba(20,184,166,0.10)'; e.target.style.background = 'rgba(20,184,166,0.04)' }}
                    onBlur={e => { e.target.style.borderColor = 'rgba(255,255,255,0.09)'; e.target.style.boxShadow = 'none'; e.target.style.background = 'rgba(255,255,255,0.04)' }}
                  />
                  <button onClick={() => send()} disabled={loading || !input.trim()}
                    style={{ width: 38, height: 38, borderRadius: 10, background: input.trim() && !loading ? 'linear-gradient(135deg,#14b8a6,#0d9488)' : 'rgba(255,255,255,0.05)', border: input.trim() && !loading ? 'none' : '1px solid rgba(255,255,255,0.07)', cursor: input.trim() && !loading ? 'pointer' : 'not-allowed', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.15s', flexShrink: 0, boxShadow: input.trim() && !loading ? '0 4px 12px rgba(20,184,166,0.30)' : 'none' }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={input.trim() && !loading ? 'white' : '#475569'} strokeWidth="2.2" strokeLinecap="round"><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></svg>
                  </button>
                </div>
                <p style={{ fontSize: 10.5, color: '#334155', textAlign: 'center', marginTop: 7 }}>Not a substitute for professional medical advice</p>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── FAB ── */}
      <button onClick={() => setOpen(o => !o)} aria-label="Open AI assistant"
        style={{
          position: 'fixed', bottom, right: 24, zIndex: 9990,
          height: 50, paddingLeft: 13, paddingRight: 18,
          borderRadius: 14,
          background: open
            ? 'rgba(17,24,39,0.95)'
            : 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)',
          border: open ? '1px solid rgba(255,255,255,0.10)' : 'none',
          cursor: 'pointer',
          display: 'flex', alignItems: 'center', gap: 10,
          boxShadow: open
            ? '0 4px 20px rgba(0,0,0,0.35)'
            : '0 4px 24px rgba(20,184,166,0.42), 0 1px 6px rgba(0,0,0,0.20)',
          transition: 'all 0.25s cubic-bezier(0.34,1.56,0.64,1)',
        }}
        onMouseEnter={e => (e.currentTarget.style.transform = 'translateY(-2px)')}
        onMouseLeave={e => (e.currentTarget.style.transform = '')}
      >
        {open ? (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2.2" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
            <span style={{ fontSize: 13.5, fontWeight: 600, color: '#94a3b8', letterSpacing: '-0.2px' }}>Close</span>
          </>
        ) : (
          <>
            <div style={{ width: 27, height: 27, borderRadius: 8, background: 'rgba(255,255,255,0.18)', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              <SparklesIcon size={13} color="white" strokeWidth={2} />
              <div style={{ position: 'absolute', top: -2, right: -2, width: 7, height: 7, borderRadius: '50%', background: '#4ade80', border: '1.5px solid #0d9488', animation: 'pulse-dot 2s infinite' }} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: 'white', lineHeight: 1.1, letterSpacing: '-0.2px' }}>Ask SAM</span>
              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.75)', fontWeight: 500 }}>Medical AI</span>
            </div>
          </>
        )}
      </button>

      <style>{`
        @keyframes bounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-4px); } }
        @keyframes chatOpen { from { opacity: 0; transform: translateY(12px) scale(0.97); } to { opacity: 1; transform: translateY(0) scale(1); } }
      `}</style>
    </>
  )
}
