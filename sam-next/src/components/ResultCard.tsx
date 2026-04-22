'use client'
import { useEffect, useRef } from 'react'
import { motion, useInView, useMotionValue, useSpring } from 'framer-motion'

interface ResultCardProps {
  prediction: number
  probability: number
  riskLevel: 'Low' | 'Moderate' | 'High'
  modelVersion?: string
  title?: string
  positiveLabel?: string
  negativeLabel?: string
  onReset?: () => void
}

function AnimatedNumber({ value }: { value: number }) {
  const ref = useRef<HTMLSpanElement>(null)
  const mv = useMotionValue(0)
  const spring = useSpring(mv, { stiffness: 80, damping: 20 })

  useEffect(() => { mv.set(value) }, [value, mv])
  useEffect(() => spring.on('change', (v) => { if (ref.current) ref.current.textContent = v.toFixed(1) }), [spring])

  return <span ref={ref}>0.0</span>
}

const riskColors: Record<string, { bg: string; text: string; bar: string; glow: string }> = {
  Low: { bg: '#f0fdf4', text: '#15803d', bar: '#22c55e', glow: 'rgba(34,197,94,0.2)' },
  Moderate: { bg: '#fffbeb', text: '#b45309', bar: '#f59e0b', glow: 'rgba(245,158,11,0.2)' },
  High: { bg: '#fef2f2', text: '#dc2626', bar: '#ef4444', glow: 'rgba(239,68,68,0.2)' },
}

export default function ResultCard({ prediction, probability, riskLevel, modelVersion, title = 'Analysis Result', positiveLabel = 'Positive', negativeLabel = 'Negative', onReset }: ResultCardProps) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })
  const colors = riskColors[riskLevel] || riskColors.Low

  return (
    <motion.div
      ref={ref}
      className="bg-white rounded-2xl shadow-lg border border-zinc-100 p-8 max-w-md w-full mx-auto"
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      <div className="flex items-start justify-between mb-6">
        <div>
          <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wide mb-1">{title}</p>
          <h2 className="text-2xl font-bold text-zinc-900">
            {prediction === 1 ? positiveLabel : negativeLabel}
          </h2>
        </div>
        <motion.div
          className="px-3 py-1.5 rounded-full text-xs font-bold border"
          style={{ background: colors.bg, color: colors.text, borderColor: colors.bar + '40', boxShadow: `0 0 12px ${colors.glow}` }}
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.3, type: 'spring', stiffness: 200 }}
        >
          {riskLevel} Risk
        </motion.div>
      </div>

      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-zinc-500 font-medium">Confidence</span>
          <span className="text-sm font-bold text-zinc-900">
            {inView ? <><AnimatedNumber value={probability} />%</> : '0.0%'}
          </span>
        </div>
        <div className="h-2.5 bg-zinc-100 rounded-full overflow-hidden">
          <motion.div
            className="h-full rounded-full"
            style={{ background: `linear-gradient(90deg, ${colors.bar}99, ${colors.bar})` }}
            initial={{ width: 0 }}
            animate={inView ? { width: `${probability}%` } : { width: 0 }}
            transition={{ duration: 1.2, delay: 0.4, ease: 'easeOut' }}
          />
        </div>
      </div>

      {modelVersion && (
        <p className="text-xs text-zinc-400 mb-6">Model v{modelVersion}</p>
      )}

      {onReset && (
        <motion.button
          onClick={onReset}
          className="w-full py-2.5 rounded-xl text-sm font-semibold border border-zinc-200 text-zinc-700 hover:bg-zinc-50 transition-colors"
          whileHover={{ y: -1 }}
          whileTap={{ scale: 0.98 }}
        >
          New Assessment
        </motion.button>
      )}
    </motion.div>
  )
}
