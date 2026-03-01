# SAM AI & Dashboard Feature Map

## Current Objective
We need to build a new set of interactive features on the `/dashboard` page for authenticated users. Currently, the dashboard only displays the same generic diagnostic module cards as the homepage. 

**Goals:**
1. Intercept and save the user's previous diagnostic assessments.
2. Display a history of these assessments in the user's dashboard.
3. Add interactive features for users to explore and play around with their health data.

---

## How to use this Context Map
**CRITICAL INSTRUCTION:** To strictly preserve token limits and run optimally, DO NOT process all of the documentation at once. 
You control the loading of information. Please only `@import` the specific files below when you absolutely need that context for the immediate task at hand.

### 1. General Overview & Tech Stack
Provides a high-level description of the Next.js App Router structure, strictly typed forms, and the custom vanilla UI styling philosophy.
`@import claude-docs/01-overview-stack.md`

### 2. Architecture & File Structure
Details the Next.js environment, global layout wrappers, and the static landing page.
`@import claude-docs/02-architecture-routing.md`

### 3. Predictive Suite & API Forms (CRITICAL FOR SAVING ASSESSMENTS)
Explains how the AI prediction suite models work, how forms trigger the Python endpoints, and how results (`ResultDisplay`) are generated. You will need this context to understand how to intercept and save assessments.
`@import claude-docs/03-predictive-suite.md`

### 4. Authentication & State (CRITICAL FOR USER BINDING)
Details the Context wrapper (`AuthProvider`) and the `RequireAuth` guard mechanisms. You will need this context to securely fetch the currently logged-in user and bind historical assessments to them.
`@import claude-docs/04-auth-context.md`

### 5. UI Components & Chat
Explains the floating chat widget assistant and the `lucide-react` medical iconography. Load this only if modifying the chat LLM API or manipulating icon visuals.
`@import claude-docs/05-ui-components.md`
