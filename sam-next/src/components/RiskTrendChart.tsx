'use client'
import { useState, useRef } from 'react'
import type { Assessment, AssessmentType } from '@/lib/assessmentHistory'

interface Props { assessments: Assessment[] }

const TYPE_COLORS: Record<AssessmentType, string> = {
  'heart':            '#f87171',
  'liver':            '#fbbf24',
  'diabetes-female':  '#a78bfa',
  'diabetes-male':    '#60a5fa',
  'breast-cancer':    '#34d399',
}

const TYPE_LABELS: Record<AssessmentType, string> = {
  'heart':           'Heart',
  'liver':           'Liver',
  'diabetes-female': 'Diabetes ♀',
  'diabetes-male':   'Diabetes ♂',
  'breast-cancer':   'Breast Cancer',
}

const W = 700, H = 240, PAD_L = 48, PAD_R = 20, PAD_T = 20, PAD_B = 36
const INNER_W = W - PAD_L - PAD_R
const INNER_H = H - PAD_T - PAD_B

function fmt(iso: string) {
  const d = new Date(iso)
  return `${d.getDate()} ${d.toLocaleString('en-GB', { month: 'short' })}`
}

export default function RiskTrendChart({ assessments }: Props) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; a: Assessment } | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  // Show last 20, oldest first
  const sorted = [...assessments].reverse().slice(0, 20)

  if (sorted.length === 0) {
    return (
      <div style={{ background: '#111827', borderRadius: 18, border: '1px solid rgba(255,255,255,0.07)', padding: '40px 24px', textAlign: 'center' }}>
        {/* Empty state grid */}
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', maxWidth: W, margin: '0 auto 16px', opacity: 0.3 }}>
          {[0, 25, 50, 75, 100].map(v => {
            const cy = PAD_T + INNER_H * (1 - v / 100)
            return (
              <g key={v}>
                <line x1={PAD_L} y1={cy} x2={W - PAD_R} y2={cy} stroke="rgba(255,255,255,0.15)" strokeWidth="1" strokeDasharray="4 4" />
                <text x={PAD_L - 6} y={cy + 4} textAnchor="end" fill="rgba(255,255,255,0.30)" fontSize="10">{v}%</text>
              </g>
            )
          })}
        </svg>
        <p style={{ fontSize: 14, color: '#475569', fontWeight: 600 }}>No assessments to trend yet</p>
        <p style={{ fontSize: 12, color: '#334155', marginTop: 4 }}>Complete a diagnostic module to see your risk trajectory here.</p>
      </div>
    )
  }

  const n = sorted.length
  const xStep = n > 1 ? INNER_W / (n - 1) : INNER_W / 2

  const px = (i: number) => PAD_L + (n === 1 ? INNER_W / 2 : i * xStep)
  const py = (prob: number) => PAD_T + INNER_H * (1 - prob / 100)

  // Group by type for polylines
  const groups: Record<string, { i: number; a: Assessment }[]> = {}
  sorted.forEach((a, i) => {
    if (!groups[a.type]) groups[a.type] = []
    groups[a.type].push({ i, a })
  })

  return (
    <div style={{ background: '#111827', borderRadius: 18, border: '1px solid rgba(255,255,255,0.07)', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '16px 20px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <p style={{ fontSize: 14, fontWeight: 700, color: '#f1f5f9' }}>Risk Trend</p>
          <p style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>Confidence % across last {n} assessment{n !== 1 ? 's' : ''}</p>
        </div>
        {/* Legend */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {(Object.keys(TYPE_COLORS) as AssessmentType[]).filter(t => groups[t]).map(t => (
            <div key={t} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: TYPE_COLORS[t] }} />
              <span style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>{TYPE_LABELS[t]}</span>
            </div>
          ))}
        </div>
      </div>

      {/* SVG Chart */}
      <div style={{ padding: '8px 0 4px', position: 'relative' }}>
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', overflow: 'visible' }}>
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map(v => {
            const cy = py(v)
            return (
              <g key={v}>
                <line x1={PAD_L} y1={cy} x2={W - PAD_R} y2={cy} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
                <text x={PAD_L - 6} y={cy + 4} textAnchor="end" fill="#334155" fontSize="10" fontFamily="inherit">{v}%</text>
              </g>
            )
          })}

          {/* X axis labels */}
          {sorted.map((a, i) => (
            <text key={i} x={px(i)} y={H - 4} textAnchor="middle" fill="#334155" fontSize="9" fontFamily="inherit">
              {fmt(a.date)}
            </text>
          ))}

          {/* Lines per type */}
          {(Object.entries(groups) as [string, { i: number; a: Assessment }[]][]).map(([type, pts]) => {
            if (pts.length < 2) return null
            const d = pts.map(({ i, a }) => `${px(i)},${py(a.probability)}`).join(' ')
            return (
              <polyline key={type}
                points={d} fill="none"
                stroke={TYPE_COLORS[type as AssessmentType]}
                strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                opacity="0.6" />
            )
          })}

          {/* Dots */}
          {sorted.map((a, i) => {
            const cx = px(i), cy = py(a.probability)
            const color = TYPE_COLORS[a.type]
            return (
              <g key={a.id}
                onMouseEnter={() => setTooltip({ x: cx, y: cy, a })}
                onMouseLeave={() => setTooltip(null)}
                style={{ cursor: 'pointer' }}>
                <circle cx={cx} cy={cy} r={tooltip?.a.id === a.id ? 7 : 5}
                  fill={color} stroke={tooltip?.a.id === a.id ? 'white' : '#111827'}
                  strokeWidth={tooltip?.a.id === a.id ? 2 : 1.5}
                  style={{ transition: 'r 0.1s' }} />
              </g>
            )
          })}

          {/* Inline tooltip */}
          {tooltip && (() => {
            const { x, y, a } = tooltip
            const tw = 160, th = 54
            const tx = Math.min(Math.max(x - tw / 2, PAD_L), W - PAD_R - tw)
            const ty = y - th - 12
            const color = TYPE_COLORS[a.type]
            return (
              <g>
                <rect x={tx} y={ty} width={tw} height={th} rx={8}
                  fill="#1e2a3a" stroke={color} strokeWidth="1" opacity="0.97" />
                <text x={tx + 10} y={ty + 17} fill="#f1f5f9" fontSize="11" fontWeight="700" fontFamily="inherit">{TYPE_LABELS[a.type]}</text>
                <text x={tx + 10} y={ty + 31} fill={color} fontSize="12" fontWeight="800" fontFamily="inherit">{a.probability.toFixed(1)}% · {a.risk_level} Risk</text>
                <text x={tx + 10} y={ty + 45} fill="#475569" fontSize="9" fontFamily="inherit">{new Date(a.date).toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })}</text>
              </g>
            )
          })()}
        </svg>
      </div>
    </div>
  )
}
