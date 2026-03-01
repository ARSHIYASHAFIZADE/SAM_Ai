# SAM AI — Overview & Tech Stack

SAM AI is a sophisticated Next.js 14/15 web application built using the App Router architecture. It provides AI-powered early disease detection using clinical machine learning models, supporting predictive modules for Heart Disease, Liver Health, Diabetes, and Breast Cancer. Wait times are reduced to seconds via machine learning endpoints. The platform also integrates an on-demand clinical AI chat assistant using Hugging Face LLMs.

## Tech Stack
* **Framework:** Next.js (App Router format), React 19 API structure.
* **Language:** TypeScript (Strict typing for forms and diagnostic results).
* **Styling:** Vanilla inline-CSS architecture augmented with Tailwind globals. Minimal reliance on heavy external UI kits; features custom UI elements for a professional enterprise feel.
* **Animations:** `framer-motion` for smooth viewport entrance, state transitions, and interactive physics.
* **3D Rendering:** `@react-three/fiber` & `@react-three/drei` for interactive hero background elements.
* **LLM Integrations:** Fetch streaming from Hugging Face Router API (`https://router.huggingface.co/v1/chat/completions`) using the `meta-llama/Llama-3.2-3B-Instruct` model.
