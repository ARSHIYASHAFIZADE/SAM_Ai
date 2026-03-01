# SAM AI — Clinical Diagnostics Application

## Overview
SAM AI is a sophisticated **Next.js 14/15** web application built over the App Router architecture, aimed at providing AI-powered early disease detection using clinical machine learning models. It supports multiple predictive modules for critical conditions such as Heart Disease, Liver Health, Diabetes, and Breast Cancer. Wait times are reduced to mere seconds via machine learning endpoints. Additionally, the platform is integrated with an on-demand clinical AI chat assistant powered by Hugging Face's LLMs.

## Tech Stack
* **Framework**: Next.js (App Router format), React 19 API structure.
* **Language**: TypeScript (Strict typing for forms and diagnostic results).
* **Styling**: Vanilla inline-CSS architecture augmented with Tailwind globals. Minimal reliance on heavy external UI kits; entirely custom UI elements to maximize the "stunning" enterprise feel.
* **Animations**: `framer-motion` for buttery smooth viewport entrance, state transitions, and interactive physics.
* **3D Rendering**: `@react-three/fiber` & `@react-three/drei` for interactive hero background elements.
* **Iconography**: `lucide-react` directly implemented into abstracted wrapper components for high quality and professional uniformity.
* **LLM Integrations**: Fetch streaming from Hugging Face Router API (`https://router.huggingface.co/v1/chat/completions`) using the `meta-llama/Llama-3.2-3B-Instruct` model.

---

## Directory Structure
```text
sam-next/
├── src/
│   ├── app/                      # Next.js App Router (Pages & APIs)
│   │   ├── layout.tsx            # Global HTML layout wrapper, font injection, and Global Providers
│   │   ├── page.tsx              # Main Landing Page / Marketing UI
│   │   ├── globals.css           # Global theme constants & basic Tailwind injection
│   │   ├── about/                # About the enterprise page
│   │   ├── dashboard/            # User Dashboard view (post-login)
│   │   ├── login/ & register/    # Authentication gateways
│   │   ├── predict/              # The Machine Learning Diagnostics Suite
│   │   │   ├── breast-cancer/    # 30-parameter FNA cytology model
│   │   │   ├── diabetes-female/  # 8-parameter PIMA metabolic risk model
│   │   │   ├── diabetes-male/    # 16-parameter symptom-based model
│   │   │   ├── heart/            # 13-parameter cardiovascular ECG/vital model
│   │   │   └── liver/            # Hepatic enzyme analysis model
│   │   └── api/
│   │       └── chat/             # Internal Next.js Edge Route proxying Hugging Face LLM keys
│   ├── components/               # Abstracted React UI / Logic
│   │   ├── AuthProvider.tsx      # React Context bridging frontend state & Private Routing mechanism
│   │   ├── ChatWidget.tsx        # Floating AI Assistant overlay with dynamic scroll-listener physics
│   │   ├── Hero3D.tsx            # The interactive 3D component on the landing hero
│   │   ├── MedicalIcons.tsx      # Pre-configured thin-stroke clinical SVG icon mappings
│   │   ├── Navbar.tsx            # Global transparent/glassmorphic navigation
│   │   └── ResultCard.tsx        # Universal diagnostic output card with probability bars
```

---

## Technical Deep-Dives by Domain

### 1. App Router & Root Setup (`src/app/layout.tsx`)
The `RootLayout` component envelops the `children` within a global `<AuthProvider>` (managing global logged-in state) and drops the `<ChatWidget />` outside the main routing tree to ensure the floating assistant remains universally accessible regardless of page transitions. It leverages Next.js `Inter` font for professional typography.

### 2. The Landing Page Architecture (`src/app/page.tsx`)
The `HomePage` is an elaborately designed static composition optimized with `framer-motion`:
* **Hero**: Contains a split desktop view. Left side: Value proposition and CTA. Right side: dynamic 3D WebGL background via `<Hero3D />` dynamically imported with `ssr: false` to avoid hydration issues on canvas elements.
* **Modules Board**: Iterates through a structural dictionary of AI prediction capabilities (Cardiology, Hepatology, Oncology) routing users to the `/predict/*` endpoints. It maps each specialty to the `lucide-react` mappings inside `MedicalIcons.tsx`. 
* **Design Philosophy**: The components lack dividing `borderTop` outlines to replicate modern premium apps (like ChatGPT/Vercel) and uses strictly neutral `zinc-900` (`#18181b`) shades across elements instead of bright primitive colors.

### 3. AI Predictive Suite (`src/app/predict/*/page.tsx`)
All predictive modules follow a specialized, unified logic setup taking `Heart Disease` as the archetype:
* Users are presented with a heavy React State form mapping complex clinical variables based on a `FIELDS` dictionary.
* Forms are strictly validated natively using mapped `<input type="number">` or `<select>` HTML primitives styled elegantly.
* On submission, Next.js calls out to an external Python/Flask endpoint mapped via `process.env.NEXT_PUBLIC_API_BASE_URL! + '/detect_heart'`.
* The external API executes a `.predict_proba()` against a trained model, returning `prediction`, `probability`, and `risk_level`.
* A local `<ResultDisplay />` component renders a fluid `framer-motion` sliding progression bar revealing the model's confidence logic dynamically.

### 4. Floating LLM Assistant (`src/components/ChatWidget.tsx`)
A highly polished bottom-right floating AI chat widget utilizing React constraints:
* Holds an array of `Message` history locally.
* **Animation & Rendering**: Uses local `@keyframes` and robust CSS `animation` properties to snap-open. 
* **Edge Routing (`/api/chat/route.ts`)**: To avoid exposing the `HF_TOKEN`, the widget POSTs pure text to Next.js's own internal backend route. That backend uses an Edge Runtime layout:
    ```typescript
    const HF_URL = 'https://router.huggingface.co/v1/chat/completions'
    ```
    This bridges the user prompt seamlessly to LLaMA 3.2 3B Instruct on Hugging Face infrastructure using the OpenAI chat-completions formatting syntax (`messages: [...]`).
* **Physics Fix**: Contains a `useEffect` hooking to `window.addEventListener('scroll')`. It calculates distance from the `document.body.offsetHeight`. If the user scrolls completely down to the document's footer, the entire widget intelligently translates `Y` by `50px` to avoid cutting off essential static page bottom data.

### 5. Private Routing & Authentication (`src/components/AuthProvider.tsx`)
Currently provides a robust Context wrapper intercepting routes:
* Exposes `useAuth()` hook throughout the codebase providing `isAuthenticated` flags.
* Uses local Storage / token mechanisms (or mocks them cleanly out-of-the-box).
* Specifically exports a wrapper component `RequireAuth` which drops directly onto the page component (e.g., `export default function HeartPage() { return <RequireAuth><Page /></RequireAuth> }`) redirecting unsigned users gracefully to `/login`.
