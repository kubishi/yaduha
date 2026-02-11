import { apiFetch } from './client'
import type { TranslateRequest, AgenticTranslateRequest, Translation } from '../types/api'

export function translatePipeline(body: TranslateRequest, apiKey?: string) {
  const headers: Record<string, string> = {}
  if (apiKey) headers['x-api-key'] = apiKey
  return apiFetch<Translation>('/translate/pipeline', {
    method: 'POST',
    body: JSON.stringify(body),
    headers,
  })
}

export function translateAgentic(body: AgenticTranslateRequest, apiKey?: string) {
  const headers: Record<string, string> = {}
  if (apiKey) headers['x-api-key'] = apiKey
  return apiFetch<Translation>('/translate/agentic', {
    method: 'POST',
    body: JSON.stringify(body),
    headers,
  })
}
