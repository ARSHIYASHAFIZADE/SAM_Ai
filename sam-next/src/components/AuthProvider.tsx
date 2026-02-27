'use client'
import { createContext, useContext, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'

const API = process.env.NEXT_PUBLIC_API_BASE_URL!

interface AuthCtx { isAuthenticated: boolean; login: () => void; logout: () => void; checked: boolean }
const Ctx = createContext<AuthCtx>({ isAuthenticated: false, login: () => {}, logout: () => {}, checked: false })

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setAuth] = useState(false)
  const [checked, setChecked] = useState(false)

  useEffect(() => {
    fetch(`${API}/@me`, { credentials: 'include' })
      .then((r) => setAuth(r.ok))
      .catch(() => setAuth(false))
      .finally(() => setChecked(true))
  }, [])

  const login = () => setAuth(true)
  const logout = () => {
    fetch(`${API}/logout`, { method: 'POST', credentials: 'include' }).finally(() => setAuth(false))
  }

  return <Ctx.Provider value={{ isAuthenticated, login, logout, checked }}>{children}</Ctx.Provider>
}

export const useAuth = () => useContext(Ctx)

export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, checked } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (checked && !isAuthenticated) router.replace('/login')
  }, [checked, isAuthenticated, router])

  if (!checked) return null
  if (!isAuthenticated) return null
  return <>{children}</>
}
