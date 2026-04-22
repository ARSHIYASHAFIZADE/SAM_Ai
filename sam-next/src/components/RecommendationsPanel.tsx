'use client'
import { motion } from 'framer-motion'
import { getRecommendations, type RecommendationItem } from '@/lib/recommendations'
import type { AssessmentType } from '@/lib/assessmentHistory'
import {
  HeartECGIcon, CalendarIcon, ActivityIcon, ShieldCheckIcon,
  AlertTriangleIcon, CheckCircleIcon, ZapIcon, BookOpenIcon, TargetIcon,
} from '@/components/MedicalIcons'

const ICON_MAP = {
  heart:    HeartECGIcon,
  calendar: CalendarIcon,
  activity: ActivityIcon,
  shield:   ShieldCheckIcon,
  alert:    AlertTriangleIcon,
  check:    CheckCircleIcon,
  zap:      ZapIcon,
  book:     BookOpenIcon,
  target:   TargetIcon,
} as const

interface Props { type: AssessmentType; risk_level: string }

export default function RecommendationsPanel({ type, risk_level }: Props) {
  const items = getRecommendations(type, risk_level)
  if (!items.length) return null

  const isHigh = risk_level === 'High'
  const isMod  = risk_level === 'Moderate'

  const headerColor = isHigh ? '#f87171' : isMod ? '#fbbf24' : '#34d399'
  const headerBg    = isHigh ? 'rgba(248,113,113,0.08)' : isMod ? 'rgba(251,191,36,0.08)' : 'rgba(52,211,153,0.08)'
  const headerBorder = isHigh ? 'rgba(248,113,113,0.20)' : isMod ? 'rgba(251,191,36,0.20)' : 'rgba(52,211,153,0.20)'

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      style={{ marginTop: 24 }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '14px 20px', borderRadius: '14px 14px 0 0', background: headerBg, border: `1px solid ${headerBorder}`, borderBottom: 'none' }}>
        <div style={{ width: 28, height: 28, borderRadius: 8, background: `rgba(${isHigh ? '248,113,113' : isMod ? '251,191,36' : '52,211,153'},0.15)`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {isHigh
            ? <AlertTriangleIcon size={14} color={headerColor} strokeWidth={2.2} />
            : <CheckCircleIcon size={14} color={headerColor} strokeWidth={2.2} />
          }
        </div>
        <div>
          <p style={{ fontSize: 13, fontWeight: 700, color: headerColor }}>Clinical Recommendations</p>
          <p style={{ fontSize: 11, color: '#64748b', marginTop: 1 }}>{risk_level} Risk · {items.length} action items</p>
        </div>
      </div>

      {/* Items */}
      <div style={{ background: '#111827', border: `1px solid ${headerBorder}`, borderTop: 'none', borderRadius: '0 0 14px 14px', overflow: 'hidden' }}>
        {items.map((item: RecommendationItem, i: number) => {
          const IconComp = ICON_MAP[item.iconName]
          const urgent = item.urgent
          return (
            <div key={i} style={{
              display: 'flex', gap: 14, padding: '16px 20px',
              borderBottom: i < items.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none',
              background: urgent ? 'rgba(248,113,113,0.04)' : 'transparent',
            }}>
              <div style={{ width: 34, height: 34, borderRadius: 10, background: urgent ? 'rgba(248,113,113,0.12)' : 'rgba(20,184,166,0.08)', border: `1px solid ${urgent ? 'rgba(248,113,113,0.25)' : 'rgba(20,184,166,0.15)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginTop: 2 }}>
                <IconComp size={16} color={urgent ? '#f87171' : '#14b8a6'} strokeWidth={1.8} />
              </div>
              <div>
                <p style={{ fontSize: 13, fontWeight: 700, color: urgent ? '#f87171' : '#f1f5f9', marginBottom: 4 }}>
                  {urgent && <span style={{ fontSize: 10, fontWeight: 800, background: 'rgba(248,113,113,0.15)', color: '#f87171', padding: '2px 7px', borderRadius: 4, marginRight: 8, letterSpacing: 0.5 }}>URGENT</span>}
                  {item.title}
                </p>
                <p style={{ fontSize: 13, color: '#64748b', lineHeight: 1.7 }}>{item.body}</p>
              </div>
            </div>
          )
        })}

        {/* Footer disclaimer */}
        <div style={{ padding: '12px 20px', background: 'rgba(255,255,255,0.02)', borderTop: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', gap: 8 }}>
          <ShieldCheckIcon size={12} color="#475569" strokeWidth={2} />
          <p style={{ fontSize: 11, color: '#475569', lineHeight: 1.5 }}>
            SAM AI is for educational purposes only. These recommendations do not constitute medical advice. Always consult a qualified healthcare professional.
          </p>
        </div>
      </div>
    </motion.div>
  )
}
