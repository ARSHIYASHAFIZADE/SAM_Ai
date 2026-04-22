'use client'
import { useState, useRef, useEffect, useCallback } from 'react'
import ReactDOM from 'react-dom'
import type { TooltipData } from '@/lib/fieldTooltips'

interface Props { data: TooltipData | null }

export default function FieldTooltip({ data }: Props) {
  const [visible, setVisible] = useState(false)
  const [mounted, setMounted] = useState(false)
  const [pos, setPos] = useState({ top: 0, left: 0, above: true, arrowLeft: 130 })
  const buttonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => setMounted(true), [])

  const calcPos = useCallback(() => {
    if (!buttonRef.current) return
    const rect = buttonRef.current.getBoundingClientRect()
    const above = rect.top >= 220
    const rawLeft = rect.left + rect.width / 2 - 130
    const left = Math.min(Math.max(rawLeft, 10), window.innerWidth - 270)
    const arrowLeft = Math.min(Math.max(rect.left + rect.width / 2 - left, 15), 245)
    const top = above ? rect.top - 8 : rect.bottom + 8
    setPos({ top, left, above, arrowLeft })
  }, [])

  useEffect(() => {
    if (!visible) return
    calcPos()
    window.addEventListener('scroll', calcPos, true)
    window.addEventListener('resize', calcPos)
    return () => {
      window.removeEventListener('scroll', calcPos, true)
      window.removeEventListener('resize', calcPos)
    }
  }, [visible, calcPos])

  useEffect(() => {
    function handler(e: MouseEvent) {
      const tooltipEl = document.getElementById('sam-tooltip-portal')
      if (
        buttonRef.current?.contains(e.target as Node) ||
        tooltipEl?.contains(e.target as Node)
      ) return
      setVisible(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  if (!data) return null

  const arrowAbove = {
    bottom: -5, borderTop: 'none' as const, borderLeft: 'none' as const,
  }
  const arrowBelow = {
    top: -5, borderBottom: 'none' as const, borderRight: 'none' as const,
  }

  const portal = mounted && visible ? ReactDOM.createPortal(
    <>
      <style>{`
        @keyframes tt-above { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes tt-below { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
      {/* Outer div: pure positioning, no animation (avoids transform conflict) */}
      <div
        id="sam-tooltip-portal"
        style={{
          position: 'fixed',
          top: pos.top,
          left: pos.left,
          transform: pos.above ? 'translateY(-100%)' : 'translateY(0)',
          zIndex: 9999,
          pointerEvents: 'auto',
        }}
      >
        {/* Inner div: animation only (opacity + directional slide) */}
        <div style={{
          width: 260,
          background: '#1a2236',
          border: '1px solid rgba(20,184,166,0.25)',
          borderRadius: 14,
          padding: '14px 16px',
          boxShadow: '0 16px 48px rgba(0,0,0,0.60), 0 0 0 1px rgba(20,184,166,0.10)',
          animation: pos.above ? 'tt-above 0.18s ease forwards' : 'tt-below 0.18s ease forwards',
          marginBottom: pos.above ? 8 : 0,
          marginTop: pos.above ? 0 : 8,
        }}>
          <div style={{
            position: 'absolute',
            left: pos.arrowLeft,
            transform: 'translateX(-50%) rotate(45deg)',
            width: 10, height: 10,
            background: '#1a2236',
            border: '1px solid rgba(20,184,166,0.25)',
            ...(pos.above ? arrowAbove : arrowBelow),
          }} />
          <p style={{ fontSize: 12, color: '#f1f5f9', lineHeight: 1.6, marginBottom: 8 }}>{data.description}</p>
          <div style={{ padding: '6px 10px', borderRadius: 8, background: 'rgba(20,184,166,0.08)', border: '1px solid rgba(20,184,166,0.15)', marginBottom: 8 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: '#14b8a6', letterSpacing: 0.3 }}>RANGE </span>
            <span style={{ fontSize: 11, color: '#94a3b8' }}>{data.range}</span>
          </div>
          <p style={{ fontSize: 11, color: '#64748b', lineHeight: 1.5, fontStyle: 'italic' }}>{data.relevance}</p>
        </div>
      </div>
    </>,
    document.body
  ) : null

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setVisible(v => !v)}
        aria-label="Field info"
        style={{
          width: 16, height: 16, borderRadius: '50%',
          border: '1.5px solid rgba(20,184,166,0.40)',
          background: visible ? 'rgba(20,184,166,0.18)' : 'rgba(20,184,166,0.08)',
          color: '#14b8a6', cursor: 'pointer', padding: 0,
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 10, fontWeight: 800, lineHeight: 1,
          transition: 'all 0.15s', flexShrink: 0,
          marginLeft: 6, verticalAlign: 'middle',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = 'rgba(20,184,166,0.18)'; e.currentTarget.style.borderColor = 'rgba(20,184,166,0.70)' }}
        onMouseLeave={e => { if (!visible) { e.currentTarget.style.background = 'rgba(20,184,166,0.08)'; e.currentTarget.style.borderColor = 'rgba(20,184,166,0.40)' } }}
      >
        i
      </button>
      {portal}
    </>
  )
}
