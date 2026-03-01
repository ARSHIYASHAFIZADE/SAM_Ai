interface FieldRule {
  key: string
  options?: unknown[]
  min?: number
  max?: number
}

export function validateForm(
  form: Record<string, string>,
  fields: FieldRule[]
): Record<string, string> {
  const errors: Record<string, string> = {}
  for (const f of fields) {
    const val = form[f.key]
    if (f.options) {
      if (!val) errors[f.key] = 'Please select an option'
    } else {
      if (val === '' || val === undefined) {
        errors[f.key] = 'This field is required'
      } else {
        const num = Number(val)
        if (isNaN(num)) {
          errors[f.key] = 'Please enter a valid number'
        } else if (f.min !== undefined && num < f.min) {
          errors[f.key] = `Must be at least ${f.min}`
        } else if (f.max !== undefined && num > f.max) {
          errors[f.key] = `Must be at most ${f.max}`
        }
      }
    }
  }
  return errors
}
