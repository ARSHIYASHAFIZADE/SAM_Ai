export type AssessmentType = 'heart' | 'liver' | 'diabetes-female' | 'diabetes-male' | 'breast-cancer'

export interface Assessment {
  id: string
  type: AssessmentType
  title: string
  date: string
  prediction: number
  probability: number
  risk_level: string
}

const KEY = 'sam_assessment_history'
const MAX = 50

export function saveAssessment(entry: Omit<Assessment, 'id' | 'date'>) {
  const all = getAssessments()
  const next: Assessment = { ...entry, id: crypto.randomUUID(), date: new Date().toISOString() }
  const updated = [next, ...all].slice(0, MAX)
  try { localStorage.setItem(KEY, JSON.stringify(updated)) } catch { /* storage full */ }
}

export function getAssessments(): Assessment[] {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as Assessment[]) : []
  } catch { return [] }
}

export function deleteAssessment(id: string) {
  localStorage.setItem(KEY, JSON.stringify(getAssessments().filter(a => a.id !== id)))
}

export function clearAssessments() {
  localStorage.removeItem(KEY)
}
