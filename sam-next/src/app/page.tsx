'use client'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import { useAuth } from '@/components/AuthProvider'
import {
  HeartECGIcon, LiverIcon, GlucoseDropIcon, SyringeIcon, MicroscopeIcon,
  ActivityIcon, SparklesIcon, ShieldCheckIcon, ZapIcon, BrainSparkIcon,
  ArrowRightIcon, CheckCircleIcon, StethoscopeIcon,
} from '@/components/MedicalIcons'

const Hero3D = dynamic(() => import('@/components/Hero3D'), { ssr: false })

const modules = [
  { title: 'Heart Disease', desc: 'Cardiovascular risk from 13 clinical vitals — ECG patterns, cholesterol, blood pressure, and angina indicators.', href: '/predict/heart', accent: 'rgba(248,113,113,0.12)', border: 'rgba(248,113,113,0.30)', iconColor: '#f87171', tag: 'Cardiology', Icon: HeartECGIcon },
  { title: 'Liver Health', desc: 'Hepatic enzyme panel — bilirubin, ALT, AST, albumin, and A/G ratio for comprehensive hepatic risk scoring.', href: '/predict/liver', accent: 'rgba(251,191,36,0.12)', border: 'rgba(251,191,36,0.30)', iconColor: '#fbbf24', tag: 'Hepatology', Icon: LiverIcon },
  { title: 'Diabetes Female', desc: 'Metabolic risk from 8 PIMA markers: glucose, BMI, insulin resistance, and diabetic pedigree function.', href: '/predict/diabetes-female', accent: 'rgba(167,139,250,0.12)', border: 'rgba(167,139,250,0.30)', iconColor: '#a78bfa', tag: 'Endocrinology', Icon: GlucoseDropIcon },
  { title: 'Diabetes Male', desc: 'Symptom driven screening across 16 clinical and lifestyle indicators including polyuria, polydipsia, and neuropathy.', href: '/predict/diabetes-male', accent: 'rgba(96,165,250,0.12)', border: 'rgba(96,165,250,0.30)', iconColor: '#60a5fa', tag: 'Endocrinology', Icon: SyringeIcon },
  { title: 'Breast Cancer', desc: '30 FNA cytology biomarkers — nucleus size, texture, compactness, and concavity for malignant vs benign classification.', href: '/predict/breast-cancer', accent: 'rgba(52,211,153,0.12)', border: 'rgba(52,211,153,0.30)', iconColor: '#34d399', tag: 'Oncology', Icon: MicroscopeIcon },
]

const features = [
  { Icon: ZapIcon, title: 'Instant Results', desc: 'ML models return risk scores and confidence intervals in under a second.' },
  { Icon: ShieldCheckIcon, title: 'Session-Scoped Privacy', desc: 'No health data is persisted server-side. All predictions are stateless.' },
  { Icon: BrainSparkIcon, title: 'Clinically Trained Models', desc: 'Gradient Boosting and Logistic Regression models trained on validated clinical datasets.' },
  { Icon: ActivityIcon, title: 'Confidence Scoring', desc: 'Every result includes a probability score and tiered risk level — Low, Moderate, or High.' },
]

export default function HomePage() {
  const { isAuthenticated } = useAuth()
  const router = useRouter()

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#0a0f1e' }}>
      <Navbar />

      <main style={{ flex: 1 }}>

        {/* ── HERO ── */}
        <section style={{ position: 'relative', overflow: 'hidden', minHeight: 'calc(100vh - 64px)', display: 'flex', alignItems: 'center' }}>
          <div className="blob" style={{ width: 800, height: 800, background: 'radial-gradient(circle, rgba(20,184,166,0.14) 0%, transparent 65%)', top: -300, right: -200 }} />
          <div className="blob" style={{ width: 500, height: 500, background: 'radial-gradient(circle, rgba(96,165,250,0.08) 0%, transparent 65%)', bottom: -100, left: -100 }} />
          <div className="blob" style={{ width: 300, height: 300, background: 'radial-gradient(circle, rgba(167,139,250,0.07) 0%, transparent 65%)', top: '40%', left: '42%' }} />

          <div style={{ position: 'relative', zIndex: 10, maxWidth: 1200, margin: '0 auto', padding: '80px 24px', width: '100%', display: 'flex', alignItems: 'center', gap: 60, flexWrap: 'wrap' }}>

            <motion.div style={{ flex: 1, minWidth: 320, maxWidth: 580 }}
              initial={{ opacity: 0, y: 32 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: 'easeOut' }}>

              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '6px 16px 6px 10px', borderRadius: 100, background: 'rgba(20,184,166,0.10)', border: '1px solid rgba(20,184,166,0.25)', marginBottom: 30 }}>
                <div style={{ width: 20, height: 20, borderRadius: '50%', background: 'rgba(20,184,166,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <ActivityIcon size={11} color="#14b8a6" strokeWidth={2.5} />
                </div>
                <span style={{ fontSize: 12, fontWeight: 700, color: '#14b8a6', letterSpacing: 0.3 }}>AI-Powered Clinical Diagnostics</span>
              </motion.div>

              <motion.h1 initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.18 }}
                style={{ fontSize: 'clamp(2.6rem, 5.5vw, 4rem)', fontWeight: 800, lineHeight: 1.08, color: '#f1f5f9', marginBottom: 26, letterSpacing: '-2px' }}>
                Detect Disease<br />
                <span className="text-teal-grad">Earlier with AI</span>
              </motion.h1>

              <motion.p initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.26 }}
                style={{ fontSize: 17, lineHeight: 1.8, color: '#94a3b8', marginBottom: 40, maxWidth: 480 }}>
                SAM AI uses clinical machine learning to assist in early detection of critical conditions — heart disease, diabetes, liver disease, and cancer.
              </motion.p>

              <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.34 }}
                style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                <motion.button
                  onClick={() => document.getElementById('modules')?.scrollIntoView({ behavior: 'smooth' })}
                  style={{ padding: '14px 28px', borderRadius: 12, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontWeight: 700, fontSize: 15, border: 'none', cursor: 'pointer', boxShadow: '0 4px 24px rgba(20,184,166,0.35)', display: 'flex', alignItems: 'center', gap: 8 }}
                  whileHover={{ y: -2, boxShadow: '0 10px 36px rgba(20,184,166,0.45)' }} whileTap={{ scale: 0.97 }}>
                  <ActivityIcon size={16} color="white" strokeWidth={2.2} />
                  Start Diagnostics
                </motion.button>
                <motion.div whileHover={{ y: -1 }}>
                  <Link href="/about" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '14px 28px', borderRadius: 12, background: 'rgba(255,255,255,0.05)', color: '#94a3b8', fontWeight: 600, fontSize: 15, textDecoration: 'none', border: '1px solid rgba(255,255,255,0.10)' }}>
                    Learn More
                    <ArrowRightIcon size={14} color="#94a3b8" strokeWidth={2.5} />
                  </Link>
                </motion.div>
              </motion.div>

              {/* Stats */}
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.55 }}
                style={{ display: 'flex', gap: 36, marginTop: 52, paddingTop: 32, borderTop: '1px solid rgba(255,255,255,0.07)' }}>
                {[['5', 'AI Modules'], ['ML', 'Powered'], ['Secure', 'Sessions'], ['Free', 'Access']].map(([v, l]) => (
                  <div key={l}>
                    <div style={{ fontSize: 20, fontWeight: 800, color: '#14b8a6' }}>{v}</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 3, fontWeight: 500, letterSpacing: 0.3 }}>{l}</div>
                  </div>
                ))}
              </motion.div>
            </motion.div>

            {/* 3D */}
            <motion.div style={{ flex: 1, minWidth: 300, height: 480 }}
              initial={{ opacity: 0, scale: 0.88 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.9, delay: 0.15, ease: 'easeOut' }}>
              <Hero3D />
            </motion.div>
          </div>
        </section>

        {/* ── MODULES ── */}
        <section id="modules" style={{ padding: '110px 24px', background: '#0d1526', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ maxWidth: 1200, margin: '0 auto' }}>
            <motion.div style={{ textAlign: 'center', marginBottom: 64 }}
              initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <p className="label-upper" style={{ marginBottom: 14 }}>Diagnostic Suite</p>
              <h2 style={{ fontSize: 'clamp(1.9rem, 3.5vw, 2.8rem)', fontWeight: 800, color: '#f1f5f9', letterSpacing: '-1px', marginBottom: 14 }}>Clinical AI Modules</h2>
              <p style={{ fontSize: 16, color: '#64748b', maxWidth: 460, margin: '0 auto' }}>Five validated machine learning models — each trained on clinical-grade datasets.</p>
            </motion.div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 20 }}>
              {modules.map((m, i) => (
                <motion.div key={m.title}
                  onClick={() => router.push(isAuthenticated ? m.href : '/login')}
                  style={{ background: '#111827', border: `1px solid ${m.border}`, borderRadius: 22, padding: 28, cursor: 'pointer', position: 'relative', overflow: 'hidden', transition: 'all 0.2s' }}
                  initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.4, delay: i * 0.07 }}
                  whileHover={{ y: -5, boxShadow: `0 20px 56px rgba(0,0,0,0.35), 0 0 0 1px ${m.border}` }}>
                  {/* accent bg glow */}
                  <div style={{ position: 'absolute', top: -40, right: -40, width: 140, height: 140, borderRadius: '50%', background: m.accent, filter: 'blur(40px)', pointerEvents: 'none' }} />
                  <div style={{ position: 'relative' }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 22 }}>
                      <div style={{ width: 54, height: 54, borderRadius: 16, background: m.accent, border: `1.5px solid ${m.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: `0 0 20px ${m.accent}` }}>
                        <m.Icon size={25} color={m.iconColor} strokeWidth={1.7} />
                      </div>
                      <span style={{ fontSize: 11, fontWeight: 700, color: '#64748b', background: 'rgba(255,255,255,0.06)', padding: '4px 11px', borderRadius: 100, border: '1px solid rgba(255,255,255,0.08)', letterSpacing: 0.4 }}>{m.tag}</span>
                    </div>
                    <h3 style={{ fontSize: 17, fontWeight: 700, color: '#f1f5f9', marginBottom: 10 }}>{m.title}</h3>
                    <p style={{ fontSize: 14, color: '#64748b', lineHeight: 1.7, marginBottom: 22 }}>{m.desc}</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#14b8a6', fontSize: 13, fontWeight: 700 }}>
                      Begin Assessment
                      <ArrowRightIcon size={13} color="#14b8a6" strokeWidth={2.5} />
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* ── FEATURES ── */}
        <section style={{ padding: '110px 24px', background: '#0a0f1e' }}>
          <div style={{ maxWidth: 1100, margin: '0 auto' }}>
            <motion.div style={{ textAlign: 'center', marginBottom: 64 }}
              initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <p className="label-upper" style={{ marginBottom: 14 }}>Platform</p>
              <h2 style={{ fontSize: 'clamp(1.9rem, 3vw, 2.6rem)', fontWeight: 800, color: '#f1f5f9', letterSpacing: '-1px', marginBottom: 14 }}>Built for clinical confidence</h2>
              <p style={{ fontSize: 16, color: '#64748b', maxWidth: 440, margin: '0 auto' }}>Every design decision supports accurate, trustworthy health screening.</p>
            </motion.div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 20 }}>
              {features.map((f, i) => (
                <motion.div key={f.title}
                  style={{ background: '#111827', borderRadius: 20, padding: 30, border: '1px solid rgba(255,255,255,0.07)' }}
                  initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.09 }}>
                  <div style={{ width: 46, height: 46, borderRadius: 13, background: 'rgba(20,184,166,0.10)', border: '1px solid rgba(20,184,166,0.20)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 20 }}>
                    <f.Icon size={22} color="#14b8a6" strokeWidth={1.7} />
                  </div>
                  <h3 style={{ fontWeight: 700, fontSize: 16, color: '#f1f5f9', marginBottom: 10 }}>{f.title}</h3>
                  <p style={{ fontSize: 14, color: '#64748b', lineHeight: 1.7 }}>{f.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* ── HOW IT WORKS ── */}
        <section style={{ padding: '110px 24px', background: '#0d1526', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ maxWidth: 1000, margin: '0 auto' }}>
            <motion.div style={{ textAlign: 'center', marginBottom: 64 }}
              initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <p className="label-upper" style={{ marginBottom: 14 }}>Workflow</p>
              <h2 style={{ fontSize: 'clamp(1.9rem, 3vw, 2.6rem)', fontWeight: 800, color: '#f1f5f9', letterSpacing: '-1px', marginBottom: 14 }}>Three steps to your report</h2>
              <p style={{ fontSize: 16, color: '#64748b' }}>From registration to AI-generated risk score in under two minutes.</p>
            </motion.div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 24 }}>
              {[
                { n: '01', title: 'Create Account', desc: 'Register securely. Session data is never stored permanently on our servers.', Icon: StethoscopeIcon, color: '#14b8a6' },
                { n: '02', title: 'Enter Clinical Data', desc: 'Input biomarker values into structured, validated diagnostic forms.', Icon: ActivityIcon, color: '#60a5fa' },
                { n: '03', title: 'Receive AI Analysis', desc: 'Instant risk assessment with confidence score, risk level, and clinical recommendations.', Icon: SparklesIcon, color: '#a78bfa' },
              ].map((s, i) => (
                <motion.div key={s.n}
                  style={{ background: '#111827', borderRadius: 22, padding: 32, border: '1px solid rgba(255,255,255,0.07)', position: 'relative', overflow: 'hidden' }}
                  initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }}>
                  <div style={{ position: 'absolute', top: 20, right: 20, fontSize: 11, fontWeight: 800, color: 'rgba(255,255,255,0.08)', letterSpacing: 2 }}>{s.n}</div>
                  <div style={{ width: 48, height: 48, borderRadius: 14, background: `rgba(${s.color === '#14b8a6' ? '20,184,166' : s.color === '#60a5fa' ? '96,165,250' : '167,139,250'},0.12)`, border: `1px solid rgba(${s.color === '#14b8a6' ? '20,184,166' : s.color === '#60a5fa' ? '96,165,250' : '167,139,250'},0.25)`, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 20 }}>
                    <s.Icon size={22} color={s.color} strokeWidth={1.7} />
                  </div>
                  <h3 style={{ fontWeight: 700, fontSize: 17, color: '#f1f5f9', marginBottom: 10 }}>{s.title}</h3>
                  <p style={{ fontSize: 14, color: '#64748b', lineHeight: 1.7 }}>{s.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <section style={{ padding: '110px 24px', background: '#0a0f1e', position: 'relative', overflow: 'hidden' }}>
          <div className="blob" style={{ width: 600, height: 600, background: 'radial-gradient(circle, rgba(20,184,166,0.10) 0%, transparent 65%)', top: '50%', left: '50%', transform: 'translate(-50%,-50%)' }} />
          <motion.div style={{ maxWidth: 640, margin: '0 auto', textAlign: 'center', position: 'relative', zIndex: 1 }}
            initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 14px', borderRadius: 100, background: 'rgba(20,184,166,0.10)', border: '1px solid rgba(20,184,166,0.20)', marginBottom: 24 }}>
              <CheckCircleIcon size={12} color="#14b8a6" strokeWidth={2.5} />
              <span style={{ fontSize: 12, fontWeight: 700, color: '#14b8a6' }}>Free · No credit card required</span>
            </div>
            <h2 style={{ fontSize: 'clamp(2rem, 4vw, 3rem)', fontWeight: 800, color: '#f1f5f9', marginBottom: 18, letterSpacing: '-1.2px' }}>Begin your health journey</h2>
            <p style={{ color: '#64748b', fontSize: 16, marginBottom: 40, lineHeight: 1.8 }}>
              Create a free account and access all five clinical AI diagnostic modules instantly.
            </p>
            {!isAuthenticated && (
              <motion.div whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}>
                <Link href="/register" style={{ display: 'inline-flex', alignItems: 'center', gap: 10, padding: '16px 36px', borderRadius: 14, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontWeight: 700, fontSize: 16, textDecoration: 'none', boxShadow: '0 4px 28px rgba(20,184,166,0.35)', letterSpacing: '-0.2px' }}>
                  <ActivityIcon size={18} color="white" strokeWidth={2.2} />
                  Create Free Account
                </Link>
              </motion.div>
            )}
          </motion.div>
        </section>
      </main>

      {/* ── FOOTER ── */}
      <footer style={{ background: '#0d1526', borderTop: '1px solid rgba(255,255,255,0.06)', padding: '36px 24px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 28, height: 28, borderRadius: 8, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <ActivityIcon size={13} color="white" strokeWidth={2.5} />
            </div>
            <span style={{ fontWeight: 800, color: '#f1f5f9', fontSize: 15 }}>SAM AI</span>
          </div>
          <p style={{ fontSize: 13, color: '#475569' }}>© {new Date().getFullYear()} SAM AI — Educational use only. Not a substitute for medical advice.</p>
          <div style={{ display: 'flex', gap: 24 }}>
            {[['/', 'Home'], ['/about', 'About'], ['/dashboard', 'Dashboard']].map(([href, label]) => (
              <Link key={href} href={href} style={{ fontSize: 13, color: '#64748b', textDecoration: 'none', transition: 'color 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.color = '#94a3b8'}
                onMouseLeave={e => e.currentTarget.style.color = '#64748b'}>
                {label}
              </Link>
            ))}
          </div>
        </div>
      </footer>
    </div>
  )
}
