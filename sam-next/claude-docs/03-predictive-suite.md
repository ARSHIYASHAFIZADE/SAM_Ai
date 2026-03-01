# AI Predictive Suite (`src/app/predict/*/page.tsx`)

All predictive modules (e.g., Heart Disease, Liver Health) follow a unified logic pattern, using the Heart Disease app as the archetype:

1. **State:** Users are presented with a heavy React State form mapping complex clinical variables based on a `FIELDS` constant array dictionary.
2. **Validation:** Forms validate securely using mapped HTML primitves (`<input type="number">` or `<select>`).
3. **API Execution:** On submission, the Next.js client calls an external Python/Flask endpoint mapped via `process.env.NEXT_PUBLIC_API_BASE_URL! + '/detect_heart'`.
4. **Response:** The external API executes `.predict_proba()` against a trained ML model and returns:
   - `prediction` (e.g., 0 or 1)
   - `probability` (Confidence percentage)
   - `risk_level` (e.g., Low, Moderate, High)
5. **Display:** A local `<ResultDisplay />` component renders a fluid `framer-motion` sliding progression bar revealing the model's confidence logic dynamically.

*To intercept data for the dashboard, you must tap into the submit workflow between receiving the response from the Python API and rendering the ResultDisplay.*
