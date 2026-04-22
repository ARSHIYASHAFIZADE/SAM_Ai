'use client'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Link from 'next/link'
import Navbar from '@/components/Navbar'
import {
  HeartECGIcon, LiverIcon, GlucoseDropIcon, SyringeIcon, MicroscopeIcon,
  TargetIcon, BookOpenIcon, ShieldCheckIcon, ZapIcon, BrainSparkIcon,
  DatabaseIcon, LockIcon, GlobeIcon, CheckCircleIcon, AwardIcon,
  ActivityIcon, CpuIcon, ServerIcon, Code2Icon, LayersIcon,
  ChevronDownIcon, ArrowRightIcon,
} from '@/components/MedicalIcons'

// ── Section: Hero ───────────────────────────────────────────
function HeroSection() {
  return (
    <section style={{ position: 'relative', overflow: 'hidden', padding: '100px 24px 80px', background: '#0a0f1e' }}>
      <div className="blob" style={{ width: 700, height: 700, background: 'radial-gradient(circle, rgba(20,184,166,0.12) 0%, transparent 65%)', top: -200, right: -150 }} />
      <div className="blob" style={{ width: 400, height: 400, background: 'radial-gradient(circle, rgba(96,165,250,0.07) 0%, transparent 65%)', bottom: -50, left: -80 }} />
      <div style={{ maxWidth: 900, margin: '0 auto', position: 'relative', zIndex: 1 }}>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '5px 14px', borderRadius: 100, background: 'rgba(20,184,166,0.10)', border: '1px solid rgba(20,184,166,0.22)', marginBottom: 24 }}>
            <ActivityIcon size={11} color="#14b8a6" strokeWidth={2.5} />
            <span style={{ fontSize: 12, fontWeight: 700, color: '#14b8a6', letterSpacing: 0.3 }}>About SAM AI</span>
          </div>
          <h1 style={{ fontSize: 'clamp(2.2rem, 4.5vw, 3.6rem)', fontWeight: 800, color: '#f1f5f9', letterSpacing: '-1.5px', marginBottom: 22, lineHeight: 1.1 }}>
            AI-powered diagnostics<br />
            <span className="text-teal-grad">built for early detection</span>
          </h1>
          <p style={{ fontSize: 18, color: '#94a3b8', lineHeight: 1.8, maxWidth: 640, marginBottom: 48 }}>
            SAM AI — Smart Adaptive Medical AI — is an educational platform that leverages machine learning to assist in early detection of critical health conditions including heart disease, diabetes, liver disease, and cancer.
          </p>
          {/* Stat pills */}
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
            {[
              { v: '5', l: 'AI Modules' },
              { v: '5', l: 'Clinical Datasets' },
              { v: '100%', l: 'Stateless & Private' },
            ].map(s => (
              <div key={s.l} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 20px', borderRadius: 12, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.09)' }}>
                <span style={{ fontSize: 18, fontWeight: 800, color: '#14b8a6' }}>{s.v}</span>
                <span style={{ fontSize: 13, color: '#64748b', fontWeight: 500 }}>{s.l}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}

// ── Section: Mission ─────────────────────────────────────────
function MissionSection() {
  const pillars = [
    { Icon: TargetIcon, title: 'Early Detection', desc: 'Detecting disease earlier saves lives. SAM AI surfaces risk signals before symptoms become critical, giving users time to act.', color: '#14b8a6' },
    { Icon: GlobeIcon, title: 'Accessible AI', desc: 'Clinical grade machine learning should not be locked behind paywalls. SAM AI is entirely free and requires only a browser.', color: '#60a5fa' },
    { Icon: BookOpenIcon, title: 'Evidence Based', desc: 'Every model is trained on peer reviewed clinical datasets — Cleveland Heart, PIMA Indian, ILPD, and Wisconsin Breast Cancer.', color: '#a78bfa' },
  ]
  return (
    <section style={{ padding: '100px 24px', background: '#0d1526', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>
        <motion.div style={{ textAlign: 'center', marginBottom: 60 }} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <p className="label-upper" style={{ marginBottom: 14 }}>Mission</p>
          <h2 style={{ fontSize: 'clamp(1.8rem, 3vw, 2.6rem)', fontWeight: 800, color: '#f1f5f9', letterSpacing: '-1px', marginBottom: 12 }}>Why SAM AI exists</h2>
          <p style={{ fontSize: 15, color: '#64748b', maxWidth: 440, margin: '0 auto' }}>Three core principles guide every design and engineering decision.</p>
        </motion.div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20 }}>
          {pillars.map((p, i) => (
            <motion.div key={p.title}
              style={{ background: '#111827', borderRadius: 22, padding: 32, border: '1px solid rgba(255,255,255,0.07)' }}
              initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.09 }}>
              <div style={{ width: 50, height: 50, borderRadius: 15, background: `rgba(${p.color === '#14b8a6' ? '20,184,166' : p.color === '#60a5fa' ? '96,165,250' : '167,139,250'},0.12)`, border: `1.5px solid rgba(${p.color === '#14b8a6' ? '20,184,166' : p.color === '#60a5fa' ? '96,165,250' : '167,139,250'},0.25)`, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 22 }}>
                <p.Icon size={24} color={p.color} strokeWidth={1.7} />
              </div>
              <h3 style={{ fontWeight: 700, fontSize: 17, color: '#f1f5f9', marginBottom: 12 }}>{p.title}</h3>
              <p style={{ fontSize: 14, color: '#64748b', lineHeight: 1.75 }}>{p.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ── Section: ML Technology ───────────────────────────────────
function TechSection() {
  const models = [
    { Icon: HeartECGIcon, title: 'Heart Disease', model: 'Gradient Boosting', dataset: 'Cleveland Heart Dataset', accuracy: '87%', features: 13, color: '#f87171', border: 'rgba(248,113,113,0.25)' },
    { Icon: LiverIcon, title: 'Liver Health', model: 'Random Forest', dataset: 'ILPD (Indian Liver)', accuracy: '79%', features: 10, color: '#fbbf24', border: 'rgba(251,191,36,0.25)' },
    { Icon: GlucoseDropIcon, title: 'Diabetes Female', model: 'Logistic Regression', dataset: 'PIMA Indian Diabetes', accuracy: '81%', features: 8, color: '#a78bfa', border: 'rgba(167,139,250,0.25)' },
    { Icon: SyringeIcon, title: 'Diabetes Male', model: 'Decision Tree', dataset: 'UCI Diabetes Symptom', accuracy: '93%', features: 16, color: '#60a5fa', border: 'rgba(96,165,250,0.25)' },
    { Icon: MicroscopeIcon, title: 'Breast Cancer', model: 'Support Vector Machine', dataset: 'Wisconsin Breast Cancer', accuracy: '96%', features: 30, color: '#34d399', border: 'rgba(52,211,153,0.25)' },
  ]
  return (
    <section style={{ padding: '100px 24px', background: '#0a0f1e' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>
        <motion.div style={{ textAlign: 'center', marginBottom: 60 }} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <p className="label-upper" style={{ marginBottom: 14 }}>ML Technology</p>
          <h2 style={{ fontSize: 'clamp(1.8rem, 3vw, 2.6rem)', fontWeight: 800, color: '#f1f5f9', letterSpacing: '-1px', marginBottom: 12 }}>The models behind SAM AI</h2>
          <p style={{ fontSize: 15, color: '#64748b', maxWidth: 480, margin: '0 auto' }}>Five independently trained classifiers, each optimised for its clinical domain.</p>
        </motion.div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 18 }}>
          {models.map((m, i) => (
            <motion.div key={m.title}
              style={{ background: '#111827', borderRadius: 20, padding: 26, border: `1px solid ${m.border}`, position: 'relative', overflow: 'hidden' }}
              initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.07 }}>
              <div style={{ position: 'absolute', top: -30, right: -30, width: 120, height: 120, borderRadius: '50%', background: `rgba(${m.color === '#f87171' ? '248,113,113' : m.color === '#fbbf24' ? '251,191,36' : m.color === '#a78bfa' ? '167,139,250' : m.color === '#60a5fa' ? '96,165,250' : '52,211,153'},0.08)`, filter: 'blur(30px)', pointerEvents: 'none' }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 18 }}>
                <div style={{ width: 42, height: 42, borderRadius: 12, background: `rgba(${m.color === '#f87171' ? '248,113,113' : m.color === '#fbbf24' ? '251,191,36' : m.color === '#a78bfa' ? '167,139,250' : m.color === '#60a5fa' ? '96,165,250' : '52,211,153'},0.12)`, border: `1.5px solid ${m.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <m.Icon size={20} color={m.color} strokeWidth={1.7} />
                </div>
                <div>
                  <div style={{ fontSize: 15, fontWeight: 700, color: '#f1f5f9' }}>{m.title}</div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 1 }}>{m.model}</div>
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', borderRadius: 10, background: 'rgba(255,255,255,0.03)' }}>
                  <span style={{ fontSize: 12, color: '#64748b', fontWeight: 600 }}>Dataset</span>
                  <span style={{ fontSize: 12, color: '#94a3b8', fontWeight: 600 }}>{m.dataset}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', borderRadius: 10, background: 'rgba(255,255,255,0.03)' }}>
                  <span style={{ fontSize: 12, color: '#64748b', fontWeight: 600 }}>Accuracy</span>
                  <span style={{ fontSize: 13, fontWeight: 800, color: m.color }}>{m.accuracy}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', borderRadius: 10, background: 'rgba(255,255,255,0.03)' }}>
                  <span style={{ fontSize: 12, color: '#64748b', fontWeight: 600 }}>Features</span>
                  <span style={{ fontSize: 12, color: '#94a3b8', fontWeight: 600 }}>{m.features} inputs</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ── Section: Privacy ─────────────────────────────────────────
function PrivacySection() {
  const pillars = [
    { Icon: DatabaseIcon, title: 'No Persistent Storage', desc: 'Health inputs are processed in-memory and immediately discarded. Nothing is written to a database.' },
    { Icon: LockIcon, title: 'Session-Scoped Only', desc: 'Authentication uses server-side session cookies that expire on logout. No tokens are stored in the browser.' },
    { Icon: ShieldCheckIcon, title: 'No PII Collected', desc: 'SAM AI does not collect names, addresses, or identifiable health data. Only email is used for authentication.' },
    { Icon: ZapIcon, title: 'HTTPS Encrypted', desc: 'All communication between your browser and the API is encrypted via TLS. No plaintext transmission.' },
  ]
  return (
    <section style={{ padding: '100px 24px', background: '#0d1526', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ maxWidth: 1000, margin: '0 auto' }}>
        <motion.div style={{ textAlign: 'center', marginBottom: 60 }} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <p className="label-upper" style={{ marginBottom: 14 }}>Privacy & Security</p>
          <h2 style={{ fontSize: 'clamp(1.8rem, 3vw, 2.6rem)', fontWeight: 800, color: '#f1f5f9', letterSpacing: '-1px', marginBottom: 12 }}>Your data is yours alone</h2>
          <p style={{ fontSize: 15, color: '#64748b', maxWidth: 440, margin: '0 auto' }}>SAM AI was designed from the ground up with health data sensitivity in mind.</p>
        </motion.div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 18 }}>
          {pillars.map((p, i) => (
            <motion.div key={p.title}
              style={{ background: '#111827', borderRadius: 20, padding: 28, border: '1px solid rgba(20,184,166,0.12)' }}
              initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.09 }}>
              <div style={{ width: 44, height: 44, borderRadius: 13, background: 'rgba(20,184,166,0.10)', border: '1.5px solid rgba(20,184,166,0.22)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 18 }}>
                <p.Icon size={21} color="#14b8a6" strokeWidth={1.7} />
              </div>
              <h3 style={{ fontWeight: 700, fontSize: 15, color: '#f1f5f9', marginBottom: 10 }}>{p.title}</h3>
              <p style={{ fontSize: 13.5, color: '#64748b', lineHeight: 1.7 }}>{p.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ── Section: Tech Stack ──────────────────────────────────────
function StackSection() {
  const stack = [
    { Icon: Code2Icon, name: 'Next.js 15', desc: 'App Router, React Server Components', color: '#f1f5f9' },
    { Icon: CpuIcon, name: 'Python / Flask', desc: 'ML inference API with scikit-learn', color: '#fbbf24' },
    { Icon: BrainSparkIcon, name: 'scikit-learn', desc: 'Gradient Boosting, SVM, Logistic Regression', color: '#34d399' },
    { Icon: LayersIcon, name: 'Three.js / R3F', desc: 'Interactive 3D hero visualisation', color: '#60a5fa' },
    { Icon: ZapIcon, name: 'Framer Motion', desc: 'Production grade micro-interactions', color: '#a78bfa' },
    { Icon: ServerIcon, name: 'Groq / Llama 3.3-70B', desc: 'AI chat inference, sub-second response', color: '#94a3b8' },
  ]
  return (
    <section style={{ padding: '100px 24px', background: '#0a0f1e' }}>
      <div style={{ maxWidth: 1000, margin: '0 auto' }}>
        <motion.div style={{ textAlign: 'center', marginBottom: 56 }} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <p className="label-upper" style={{ marginBottom: 14 }}>Tech Stack</p>
          <h2 style={{ fontSize: 'clamp(1.8rem, 3vw, 2.6rem)', fontWeight: 800, color: '#f1f5f9', letterSpacing: '-1px' }}>Built with modern tooling</h2>
        </motion.div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))', gap: 14 }}>
          {stack.map((s, i) => (
            <motion.div key={s.name}
              style={{ display: 'flex', alignItems: 'center', gap: 16, background: '#111827', borderRadius: 16, padding: '18px 22px', border: '1px solid rgba(255,255,255,0.07)' }}
              initial={{ opacity: 0, x: -12 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.06 }}>
              <div style={{ width: 40, height: 40, borderRadius: 12, background: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <s.Icon size={20} color={s.color} strokeWidth={1.7} />
              </div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#f1f5f9', marginBottom: 2 }}>{s.name}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>{s.desc}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ── Section: FAQ ─────────────────────────────────────────────
function FAQSection() {
  const [open, setOpen] = useState<number | null>(null)
  const faqs = [
    { q: 'How accurate are the AI models?', a: 'Accuracy varies by module — from 79% (Liver) to 96% (Breast Cancer) — on held-out test sets. These figures reflect research-grade performance, not validated clinical deployment. Always confirm results with a licensed clinician.' },
    { q: 'Is my health data stored or shared?', a: 'No. SAM AI processes all inputs in stateless API calls. Your data is never written to a database or shared with third parties. Assessment history is stored only in your browser\'s local storage.' },
    { q: 'What ML models does SAM AI use?', a: 'SAM AI uses Gradient Boosting, Random Forest, Logistic Regression, Decision Tree, and Support Vector Machine classifiers — each selected based on performance on its respective clinical dataset.' },
    { q: 'Who should use SAM AI?', a: 'SAM AI is intended for educational and research purposes — students, healthcare educators, developers, and curious individuals. It is not intended for clinical diagnosis or to replace professional medical assessment.' },
    { q: 'Can I use SAM AI instead of seeing a doctor?', a: 'Absolutely not. SAM AI is an educational tool only. If you have concerns about your health, consult a qualified healthcare professional. SAM AI\'s outputs should never be used to make medical decisions.' },
    { q: 'How do I interpret a "High Risk" result?', a: 'A High Risk result means the model detected a pattern in your inputs consistent with positive cases in its training data. It does not confirm a diagnosis. Use it as a prompt to seek professional evaluation, not as a definitive verdict.' },
  ]
  return (
    <section style={{ padding: '100px 24px', background: '#0d1526', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ maxWidth: 760, margin: '0 auto' }}>
        <motion.div style={{ textAlign: 'center', marginBottom: 60 }} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <p className="label-upper" style={{ marginBottom: 14 }}>FAQ</p>
          <h2 style={{ fontSize: 'clamp(1.8rem, 3vw, 2.6rem)', fontWeight: 800, color: '#f1f5f9', letterSpacing: '-1px' }}>Common questions</h2>
        </motion.div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {faqs.map((f, i) => (
            <motion.div key={i}
              style={{ background: '#111827', borderRadius: 16, border: '1px solid rgba(255,255,255,0.07)', overflow: 'hidden' }}
              initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.06 }}>
              <button onClick={() => setOpen(open === i ? null : i)}
                style={{ width: '100%', padding: '20px 24px', background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, textAlign: 'left' }}>
                <span style={{ fontSize: 15, fontWeight: 600, color: open === i ? '#14b8a6' : '#f1f5f9', flex: 1 }}>{f.q}</span>
                <div style={{ transform: open === i ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.25s', flexShrink: 0 }}>
                  <ChevronDownIcon size={18} color={open === i ? '#14b8a6' : '#64748b'} strokeWidth={2} />
                </div>
              </button>
              <AnimatePresence>
                {open === i && (
                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.22 }}>
                    <div style={{ padding: '0 24px 20px', fontSize: 14, color: '#64748b', lineHeight: 1.8, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                      <div style={{ paddingTop: 16 }}>{f.a}</div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ── Section: CTA ─────────────────────────────────────────────
function CTASection() {
  return (
    <section style={{ padding: '100px 24px', background: 'linear-gradient(135deg, #0a1f1e 0%, #0d1f2e 50%, #0a0f1e 100%)', borderTop: '1px solid rgba(20,184,166,0.15)', position: 'relative', overflow: 'hidden' }}>
      <div className="blob" style={{ width: 600, height: 600, background: 'radial-gradient(circle, rgba(20,184,166,0.12) 0%, transparent 65%)', top: '50%', left: '50%', transform: 'translate(-50%,-50%)' }} />
      <motion.div style={{ maxWidth: 580, margin: '0 auto', textAlign: 'center', position: 'relative', zIndex: 1 }}
        initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
        <div style={{ width: 60, height: 60, borderRadius: 18, background: 'rgba(20,184,166,0.12)', border: '1.5px solid rgba(20,184,166,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 28px', boxShadow: '0 0 30px rgba(20,184,166,0.15)' }}>
          <AwardIcon size={28} color="#14b8a6" strokeWidth={1.7} />
        </div>
        <h2 style={{ fontSize: 'clamp(2rem, 4vw, 3rem)', fontWeight: 800, color: '#f1f5f9', marginBottom: 18, letterSpacing: '-1.2px' }}>Ready to run your first assessment?</h2>
        <p style={{ color: '#64748b', fontSize: 16, marginBottom: 40, lineHeight: 1.8 }}>Create a free account and access all five clinical AI diagnostic modules instantly.</p>
        <div style={{ display: 'flex', gap: 14, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link href="/register"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '14px 28px', borderRadius: 12, background: 'linear-gradient(135deg,#14b8a6,#0d9488)', color: 'white', fontWeight: 700, fontSize: 15, textDecoration: 'none', boxShadow: '0 4px 24px rgba(20,184,166,0.35)' }}>
            <ActivityIcon size={16} color="white" strokeWidth={2.2} />
            Start Assessment
          </Link>
          <Link href="/#modules"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '14px 28px', borderRadius: 12, background: 'rgba(255,255,255,0.05)', color: '#94a3b8', fontWeight: 600, fontSize: 15, textDecoration: 'none', border: '1px solid rgba(255,255,255,0.10)' }}>
            View Modules
            <ArrowRightIcon size={14} color="#94a3b8" strokeWidth={2.5} />
          </Link>
        </div>
      </motion.div>
    </section>
  )
}

// ── Page ─────────────────────────────────────────────────────
export default function AboutPage() {
  return (
    <div style={{ minHeight: '100vh', background: '#0a0f1e' }}>
      <Navbar />
      <HeroSection />
      <MissionSection />
      <TechSection />
      <PrivacySection />
      <StackSection />
      <FAQSection />
      <CTASection />
      {/* Footer */}
      <footer style={{ background: '#0d1526', borderTop: '1px solid rgba(255,255,255,0.06)', padding: '28px 24px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
          <span style={{ fontWeight: 800, color: '#f1f5f9', fontSize: 14 }}>SAM AI</span>
          <p style={{ fontSize: 13, color: '#475569' }}>© {new Date().getFullYear()} SAM AI — Educational use only. Not a substitute for medical advice.</p>
          <div style={{ display: 'flex', gap: 20 }}>
            {[['/', 'Home'], ['/dashboard', 'Dashboard']].map(([href, label]) => (
              <Link key={href} href={href} style={{ fontSize: 13, color: '#64748b', textDecoration: 'none' }}>{label}</Link>
            ))}
          </div>
        </div>
      </footer>
    </div>
  )
}
