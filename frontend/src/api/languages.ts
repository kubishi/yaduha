import { apiFetch } from './client'
import type { LanguageListResponse, LanguageDetail } from '../types/api'

export function fetchLanguages() {
  return apiFetch<LanguageListResponse>('/languages')
}

export function fetchLanguage(code: string) {
  return apiFetch<LanguageDetail>(`/languages/${code}`)
}
