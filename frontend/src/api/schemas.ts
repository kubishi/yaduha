import { apiFetch } from './client'
import type { SentenceSchemaResponse, SentenceExamplesResponse } from '../types/api'

export function fetchSchema(code: string, sentenceType: string) {
  return apiFetch<SentenceSchemaResponse>(
    `/languages/${code}/sentence-types/${sentenceType}/schema`,
  )
}

export function fetchExamples(code: string, sentenceType: string) {
  return apiFetch<SentenceExamplesResponse>(
    `/languages/${code}/sentence-types/${sentenceType}/examples`,
  )
}
