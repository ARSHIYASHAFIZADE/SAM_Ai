# Private Routing & Authentication Context

The application relies heavily on `src/components/AuthProvider.tsx` to secure routes.

### Mechanisms
* **Context Hook:** Exposes the `useAuth()` hook throughout the codebase which provides the `isAuthenticated` flag (and user context).
* **Guards:** Specifically exports a wrapper component `<RequireAuth>` which protects active page components.
* **Example Application:** 
  ```typescript
  export default function HeartPage() { 
      return <RequireAuth><Page /></RequireAuth> 
  }
  ```
  If an unauthenticated user attempts to view a protected route, `<RequireAuth>` redirects them gracefully to `/login`.

*When establishing the new Dashboard assessment tracking features, use `useAuth` to bind fetched database/localStorage records securely to the active session parameters.*
