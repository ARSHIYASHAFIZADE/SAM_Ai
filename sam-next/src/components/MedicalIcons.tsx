// Premium professional medical SVG icon set — thin stroke-based, strict clinical geometric style
import type { CSSProperties } from 'react'

interface IconProps { size?: number; color?: string; strokeWidth?: number; style?: CSSProperties }

export function HeartECGIcon({ size = 24, color = 'currentColor', strokeWidth = 1.6, style }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" style={style}>
      <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
      <path d="M3 12h3.5l1.5-2 2 6 2.5-9 1.5 5h4" strokeWidth={strokeWidth} strokeLinecap="square"/>
    </svg>
  )
}

export function LiverIcon({ size = 24, color = 'currentColor', strokeWidth = 1.6, style }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" style={style}>
      <path d="M3 9C3 6.23858 5.23858 4 8 4H16C18.7614 4 21 6.23858 21 9V15C21 17.7614 18.7614 20 16 20H8C5.23858 20 3 17.7614 3 15V9Z"/>
      <path d="M8 8V16"/>
      <path d="M16 8V16"/>
      <path d="M12 4V20"/>
    </svg>
  )
}

export function GlucoseDropIcon({ size = 24, color = 'currentColor', strokeWidth = 1.6, style }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" style={style}>
      <path d="M12 2.5L7.5 7.5V16C7.5 18.5 9.5 20.5 12 20.5C14.5 20.5 16.5 18.5 16.5 16V7.5L12 2.5Z"/>
      <path d="M10 16.5H14"/>
      <path d="M12 14.5V18.5"/>
    </svg>
  )
}

export function SyringeIcon({ size = 24, color = 'currentColor', strokeWidth = 1.6, style }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" style={style}>
      <path d="M18 6L14 2"/>
      <path d="M16 4L4 16L8 20L20 8L16 4Z"/>
      <path d="M8 12L12 16"/>
      <path d="M2.5 21.5L4.5 19.5"/>
      <path d="M14 6L18 10"/>
      <path d="M11 9L15 13"/>
    </svg>
  )
}

export function MicroscopeIcon({ size = 24, color = 'currentColor', strokeWidth = 1.6, style }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" style={style}>
      <path d="M6 18H4a2 2 0 0 1-2-2v-1h20v1a2 2 0 0 1-2 2h-2"/>
      <path d="M6 22h12"/>
      <path d="M8 22v-4"/>
      <path d="M16 22v-4"/>
      <path d="M9 3h6l2 7H7z"/>
      <path d="M12 10v5"/>
      <path d="M10 3V1"/>
      <path d="M14 3V1"/>
    </svg>
  )
}

export function StethoscopeIcon({ size = 24, color = 'currentColor', strokeWidth = 1.6, style }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" style={style}>
      <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
      <path d="M8 15a6 6 0 0 0 6 6H17a3 3 0 0 0 3-3v-2"/>
      <circle cx="20" cy="10" r="2"/>
    </svg>
  )
}

export function BrainSparkIcon({ size = 24, color = 'currentColor', strokeWidth = 1.6, style }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" style={style}>
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-1.66z"/>
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-1.66z"/>
    </svg>
  )
}

export function SparklesIcon({ size = 24, color = 'currentColor', strokeWidth = 1.6, style }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" style={style}>
      <path d="M12 2l2.5 7.5L22 12l-7.5 2.5L12 22l-2.5-7.5L2 12l7.5-2.5Z"/>
      <path d="M5 4l1 3 3 1-3 1-1 3-1-3-3-1 3-1Z"/>
      <path d="M19 18l.8 2.4L22.2 21.2l-2.4.8L19 24l-.8-2.4L15.8 20.8l2.4-.8Z"/>
    </svg>
  )
}

export function ActivityIcon({ size = 24, color = 'currentColor', strokeWidth = 1.6, style }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" style={style}>
      <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
    </svg>
  )
}

