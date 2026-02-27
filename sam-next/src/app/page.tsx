'use client'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { useAuth } from '@/components/AuthProvider'
import { HeartECGIcon, LiverIcon, GlucoseDropIcon, SyringeIcon, MicroscopeIcon, ActivityIcon, SparklesIcon } from '@/components/MedicalIcons'

const Hero3D = dynamic(() => import('@/components/Hero3D'), { ssr: false })

const modules = [
  {
    title: 'Heart Disease', desc: 'Cardiovascular risk assessment from 13 clinical vitals including ECG, cholesterol, and blood pressure.',
    href: '/predict/heart', accent: '#fef2f2', border: '#fca5a5', iconColor: '#ef4444', tag: 'Cardiology',
    Icon: HeartECGIcon,
  },
  {
    title: 'Liver Health', desc: 'Hepatic enzyme panel analysis — bilirubin, ALT, AST, albumin, and globulin ratio.',
    href: '/predict/liver', accent: '#fffbeb', border: '#fcd34d', iconColor: '#d97706', tag: 'Hepatology',
    Icon: LiverIcon,
  },
  {
    title: 'Diabetes (Female)', desc: 'Metabolic risk scoring using 8 PIMA markers including glucose, BMI, and insulin levels.',
    href: '/predict/diabetes-female', accent: '#fdf4ff', border: '#d8b4fe', iconColor: '#9333ea', tag: 'Endocrinology',
    Icon: GlucoseDropIcon,
  },
  {
    title: 'Diabetes (Male)', desc: 'Symptom-based diabetes screening across 16 clinical and lifestyle indicators.',
    href: '/predict/diabetes-male', accent: '#eff6ff', border: '#93c5fd', iconColor: '#2563eb', tag: 'Endocrinology',
    Icon: SyringeIcon,
  },
  {
    title: 'Breast Cancer', desc: '30 FNA cytology biomarkers for malignant vs benign cell nucleus classification.',
    href: '/predict/breast-cancer', accent: '#f0fdf4', border: '#86efac', iconColor: '#16a34a', tag: 'Oncology',
    Icon: MicroscopeIcon,
  },
]

export default function HomePage() {
  const { isAuthenticated } = useAuth()
  const router = useRouter()

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#fafafa' }}>
      <Navbar />

      <main style={{ flex: 1 }}>

        {/* ── HERO ── */}
        <section style={{ position: 'relative', overflow: 'hidden', minHeight: 'calc(100vh - 64px)', display: 'flex', alignItems: 'center' }}>
          <div className="blob" style={{ width: 700, height: 700, background: 'radial-gradient(circle, rgba(15,157,154,0.11) 0%, transparent 70%)', top: -200, right: -150 }} />
          <div className="blob" style={{ width: 400, height: 400, background: 'radial-gradient(circle, rgba(15,157,154,0.06) 0%, transparent 70%)', bottom: -100, left: -80 }} />
          <div className="blob" style={{ width: 300, height: 300, background: 'radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%)', top: '35%', left: '38%' }} />

          <div style={{ position: 'relative', zIndex: 10, maxWidth: 1200, margin: '0 auto', padding: '60px 24px', width: '100%', display: 'flex', alignItems: 'center', gap: 60, flexWrap: 'wrap' }}>
            <motion.div style={{ flex: 1, minWidth: 320, maxWidth: 560 }} initial={{ opacity: 0, y: 32 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: 'easeOut' }}>

              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '6px 14px', borderRadius: 100, background: 'rgba(15,157,154,0.08)', border: '1px solid rgba(15,157,154,0.2)', marginBottom: 28, fontSize: 12, fontWeight: 700, color: '#0F9D9A' }}>
                <ActivityIcon size={13} color="#0F9D9A" strokeWidth={2.2} />
                AI-Powered Medical Diagnostics
              </motion.div>

              <motion.h1 initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.18 }}
                style={{ fontSize: 'clamp(2.5rem, 5vw, 3.75rem)', fontWeight: 800, lineHeight: 1.1, color: '#09090b', marginBottom: 24, letterSpacing: '-1.5px' }}>
                Detect Disease<br />
                <span className="text-teal-grad">Earlier with AI</span>
              </motion.h1>

              <motion.p initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.26 }}
                style={{ fontSize: 18, lineHeight: 1.75, color: '#71717a', marginBottom: 40, maxWidth: 460 }}>
                SAM AI uses clinical machine learning to assist in early diagnosis of critical conditions — heart disease, diabetes, liver disease, and cancer.
              </motion.p>

              <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.34 }} style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                <motion.button onClick={() => document.getElementById('modules')?.scrollIntoView({ behavior: 'smooth' })}
                  style={{ padding: '14px 28px', borderRadius: 12, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: 'pointer', boxShadow: '0 4px 20px rgba(15,157,154,0.32)', letterSpacing: '-0.2px', display: 'flex', alignItems: 'center', gap: 8 }}
                  whileHover={{ y: -2, boxShadow: '0 12px 32px rgba(15,157,154,0.38)' }} whileTap={{ scale: 0.97 }}>
                  <ActivityIcon size={16} color="white" strokeWidth={2.2} />
                  Start Diagnostics
                </motion.button>
                <motion.div whileHover={{ y: -1 }}>
                  <Link href="/about" style={{ display: 'inline-block', padding: '14px 28px', borderRadius: 12, background: 'white', color: '#3f3f46', fontWeight: 600, fontSize: 15, textDecoration: 'none', border: '1px solid #e4e4e7', boxShadow: '0 1px 8px rgba(0,0,0,0.05)' }}>
                    Learn More
                  </Link>
                </motion.div>
              </motion.div>

              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.55 }}
                style={{ display: 'flex', gap: 40, marginTop: 48, paddingTop: 36, borderTop: '1px solid #f4f4f5' }}>
                {[['5', 'AI Modules'], ['ML', 'Powered'], ['Secure', 'Sessions']].map(([v, l]) => (
                  <div key={l}>
                    <div style={{ fontSize: 22, fontWeight: 800, color: '#09090b' }}>{v}</div>
                    <div style={{ fontSize: 12, color: '#a1a1aa', marginTop: 2, fontWeight: 500 }}>{l}</div>
                  </div>
                ))}
              </motion.div>
            </motion.div>

            <motion.div style={{ flex: 1, minWidth: 300, height: 480 }} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8, delay: 0.15, ease: 'easeOut' }}>
              <Hero3D />
            </motion.div>
          </div>
        </section>

        {/* ── MODULES ── */}
        <section id="modules" style={{ padding: '100px 24px', background: 'white', borderTop: '1px solid #f4f4f5' }}>
          <div style={{ maxWidth: 1200, margin: '0 auto' }}>
            <motion.div style={{ textAlign: 'center', marginBottom: 60 }} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <div style={{ display: 'inline-block', padding: '5px 14px', borderRadius: 100, background: 'rgba(15,157,154,0.07)', border: '1px solid rgba(15,157,154,0.2)', fontSize: 12, fontWeight: 700, color: '#0F9D9A', marginBottom: 16 }}>Diagnostic Suite</div>
              <h2 style={{ fontSize: 36, fontWeight: 800, color: '#09090b', letterSpacing: '-1px', marginBottom: 12 }}>Clinical AI Modules</h2>
              <p style={{ fontSize: 16, color: '#71717a', maxWidth: 440, margin: '0 auto' }}>Select a module to begin your AI-assisted health assessment.</p>
            </motion.div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 20 }}>
              {modules.map((m, i) => (
                <motion.div key={m.title} onClick={() => router.push(isAuthenticated ? m.href : '/login')}
                  style={{ background: 'white', border: '1px solid #f0f0f0', borderRadius: 20, padding: 28, cursor: 'pointer', position: 'relative', overflow: 'hidden' }}
                  initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.4, delay: i * 0.07 }}
                  whileHover={{ y: -5, boxShadow: '0 20px 48px rgba(0,0,0,0.09)', borderColor: m.border }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20 }}>
                    <div style={{ width: 52, height: 52, borderRadius: 14, background: m.accent, border: `1.5px solid ${m.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <m.Icon size={24} color={m.iconColor} strokeWidth={1.8} />
                    </div>
                    <span style={{ fontSize: 11, fontWeight: 700, color: '#71717a', background: '#f4f4f5', padding: '4px 10px', borderRadius: 100, letterSpacing: 0.3 }}>{m.tag}</span>
                  </div>
                  <h3 style={{ fontSize: 17, fontWeight: 700, color: '#09090b', marginBottom: 8 }}>{m.title}</h3>
                  <p style={{ fontSize: 14, color: '#71717a', lineHeight: 1.65, marginBottom: 20 }}>{m.desc}</p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#0F9D9A', fontSize: 13, fontWeight: 700 }}>
                    Begin Assessment
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* ── HOW IT WORKS ── */}
        <section style={{ padding: '100px 24px', background: '#fafafa', borderTop: '1px solid #f4f4f5' }}>
          <div style={{ maxWidth: 1000, margin: '0 auto' }}>
            <motion.div style={{ textAlign: 'center', marginBottom: 56 }} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <h2 style={{ fontSize: 34, fontWeight: 800, color: '#09090b', marginBottom: 12, letterSpacing: '-0.8px' }}>How it works</h2>
              <p style={{ color: '#71717a', fontSize: 16 }}>Simple, fast, and secure — three steps to your AI report</p>
            </motion.div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 24 }}>
              {[
                { n: '01', title: 'Create Account', desc: 'Register securely. Your session data is never stored permanently.', Icon: StethoscopeIconLocal },
                { n: '02', title: 'Enter Patient Data', desc: 'Input clinical measurements into the structured diagnostic form.', Icon: ActivityIcon },
                { n: '03', title: 'Get AI Analysis', desc: 'Receive instant risk assessment with confidence scores and risk levels.', Icon: SparklesIcon },
              ].map((s, i) => (
                <motion.div key={s.n} style={{ background: 'white', borderRadius: 20, padding: 32, border: '1px solid #f0f0f0' }}
                  initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }}>
                  <div style={{ fontSize: 11, fontWeight: 800, color: '#0F9D9A', letterSpacing: 2, marginBottom: 16 }}>{s.n}</div>
                  <div style={{ width: 44, height: 44, borderRadius: 12, background: 'rgba(15,157,154,0.08)', border: '1px solid rgba(15,157,154,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 18 }}>
                    <s.Icon size={22} color="#0F9D9A" strokeWidth={1.7} />
                  </div>
                  <h3 style={{ fontWeight: 700, fontSize: 17, color: '#09090b', marginBottom: 10 }}>{s.title}</h3>
                  <p style={{ fontSize: 14, color: '#71717a', lineHeight: 1.65 }}>{s.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <section style={{ padding: '100px 24px', background: 'white', borderTop: '1px solid #f4f4f5' }}>
          <motion.div style={{ maxWidth: 640, margin: '0 auto', textAlign: 'center' }} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
            <h2 style={{ fontSize: 36, fontWeight: 800, color: '#09090b', marginBottom: 16, letterSpacing: '-1px' }}>Begin your health journey</h2>
            <p style={{ color: '#71717a', fontSize: 16, marginBottom: 36, lineHeight: 1.7 }}>Create a free account and access all five clinical AI diagnostic modules.</p>
            {!isAuthenticated && (
              <motion.div whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}>
                <Link href="/register" style={{ display: 'inline-flex', alignItems: 'center', gap: 10, padding: '16px 36px', borderRadius: 14, background: 'linear-gradient(135deg,#0F9D9A,#0d7d7a)', color: 'white', fontWeight: 700, fontSize: 16, textDecoration: 'none', boxShadow: '0 4px 24px rgba(15,157,154,0.32)', letterSpacing: '-0.2px' }}>
                  <ActivityIcon size={18} color="white" strokeWidth={2.2} />
                  Create Free Account
                </Link>
              </motion.div>
            )}
          </motion.div>
        </section>
      </main>

      <footer style={{ borderTop: '1px solid #f4f4f5', background: 'white', padding: '32px 24px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
          <span style={{ fontWeight: 700, color: '#0F9D9A', fontSize: 15 }}>SAM AI</span>
          <p style={{ fontSize: 13, color: '#a1a1aa' }}>© 2024 SAM AI — Educational use only. Not a substitute for medical advice.</p>
          <div style={{ display: 'flex', gap: 20 }}>
            {[['/', 'Home'], ['/about', 'About']].map(([href, label]) => (
              <Link key={href} href={href} style={{ fontSize: 13, color: '#71717a', textDecoration: 'none' }}>{label}</Link>
            ))}
          </div>
        </div>
      </footer>
    </div>
  )
}

// Inline stethoscope for How It Works step 1 (to avoid import clash)
function StethoscopeIconLocal({ size = 24, color = 'currentColor', strokeWidth = 1.8 }: { size?: number; color?: string; strokeWidth?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
      <path d="M8 15a6 6 0 0 0 6 6H17a3 3 0 0 0 3-3v-2"/>
      <circle cx="20" cy="10" r="2"/>
    </svg>
  )
}
