export interface LanguageSummary {
  code: string
  name: string
  sentence_type_count: number
  sentence_types: string[]
}

export interface LanguageListResponse {
  languages: LanguageSummary[]
}

export interface SentenceTypeInfo {
  name: string
  field_count: number
}

export interface LanguageDetail {
  code: string
  name: string
  sentence_types: SentenceTypeInfo[]
}

export interface SentenceSchemaResponse {
  language_code: string
  sentence_type: string
  json_schema: JsonSchema
}

export interface ExamplePair {
  english: string
  structured: Record<string, unknown>
  rendered: string
}

export interface SentenceExamplesResponse {
  language_code: string
  sentence_type: string
  examples: ExamplePair[]
}

export interface AgentConfig {
  provider: string
  model: string
  temperature?: number
}

export interface TranslateRequest {
  text: string
  language_code: string
  agent: AgentConfig
  back_translation_agent?: AgentConfig
}

export interface AgenticTranslateRequest {
  text: string
  language_code: string
  agent: AgentConfig
  system_prompt?: string
}

export interface BackTranslation {
  sentence_type: string
  english: string
}

export interface Translation {
  english: string
  translation: string
  sentences: Record<string, unknown>[]
  back_translations: BackTranslation[]
  metadata: Record<string, unknown>
}

// JSON Schema types (subset we care about)
export interface JsonSchema {
  title?: string
  type?: string
  properties?: Record<string, JsonSchema>
  required?: string[]
  $defs?: Record<string, JsonSchema>
  $ref?: string
  anyOf?: JsonSchema[]
  enum?: (string | number)[]
  description?: string
  const?: unknown
  items?: JsonSchema
  default?: unknown
}
