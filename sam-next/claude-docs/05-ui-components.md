# UI Elements & The Chat Assistant

### Floating LLM Assistant (`src/components/ChatWidget.tsx`)
A highly polished bottom-right floating AI chat widget utilizing React constraints:
* **Storage:** Holds an array of `Message` history locally.
* **Edge Routing (`/api/chat/route.ts`):** To avoid exposing the `HF_TOKEN`, the widget POSTs text strictly to Next.js's internal backend route. 
  ```typescript
  const HF_URL = 'https://router.huggingface.co/v1/chat/completions'
  ```
  This securely links the user prompt to the `meta-llama/Llama-3.2-3B-Instruct` model on Hugging Face using the standard OpenAI completions schema interface.
* **Scroll Physics:** Features an intelligent `useEffect` event listener resolving `window.innerHeight + window.scrollY`. It calculates if the user hits the bottom of the document footer, successfully translating the entire widget `Y` up by `50px` to avoid cutting off static page data.

### Iconography (`src/components/MedicalIcons.tsx`)
* **Libraries:** The platform dropped standard SVG paths internally for the robust `lucide-react` library.
* **Clinical Aesthetic:** The standard is mappings pointing strictly to highly professional medical assets (`HeartPulse`, `TestTubes`, `Microscope`) to uphold enterprise fidelity.
