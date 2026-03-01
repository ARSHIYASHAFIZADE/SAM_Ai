# Directory Structure & Architecture

## Core Folders
```text
sam-next/
├── src/
│   ├── app/                      
│   │   ├── layout.tsx            # Global HTML layout wrapper & Providers
│   │   ├── page.tsx              # Main Landing Page
│   │   ├── dashboard/            # User Dashboard view (post-login)
│   │   ├── login/ & register/    # Authentication gateways
│   │   ├── predict/              # The Machine Learning Diagnostics Suite
│   │   └── api/                  # Next.js Edge Routes
│   ├── components/               # Abstracted React UI / Logic
```

## Technical Execution
### 1. App Router & Root Setup (`src/app/layout.tsx`)
The `RootLayout` component wraps the `children` within a global `<AuthProvider>` (managing global logged-in state) and places `<ChatWidget />` outside the main routing tree so the floating assistant remains accessible across page transitions. It uses the `Inter` font for professional typography.

### 2. The Landing Page Architecture (`src/app/page.tsx`)
Optimized with `framer-motion`:
* **Hero:** Split desktop view with a value proposition and 3D WebGL background (`<Hero3D />` is dynamically imported with `ssr: false`).
* **Modules Board:** Iterates through AI prediction capabilities (Cardiology, Hepatology, Oncology), routing users to the `/predict/*` endpoints. 
* **Design Philosophy:** Professional minimal design lacking horizontal dividing `borderTop` outlines. It heavily uses strictly neutral `zinc-900` (`#18181b`) shades across elements instead of bright primitive gradients.
