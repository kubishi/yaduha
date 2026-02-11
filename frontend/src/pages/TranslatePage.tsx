import { useEffect, useState } from 'react'
import { fetchLanguages } from '../api/languages'
import { translatePipeline } from '../api/translate'
import type { LanguageSummary, Translation } from '../types/api'

const PROVIDERS = [
  {
    value: 'openai',
    label: 'OpenAI',
    models: ['gpt-4o', 'gpt-4o-mini'],
  },
  {
    value: 'anthropic',
    label: 'Anthropic',
    models: ['claude-sonnet-4-5-20250929'],
  },
  {
    value: 'gemini',
    label: 'Gemini',
    models: ['gemini-2.5-flash'],
  },
  {
    value: 'ollama',
    label: 'Ollama (local)',
    models: ['llama3.1'],
  },
]

export default function TranslatePage() {
  const [languages, setLanguages] = useState<LanguageSummary[]>([])
  const [langCode, setLangCode] = useState('')
  const [provider, setProvider] = useState(PROVIDERS[0].value)
  const [model, setModel] = useState(PROVIDERS[0].models[0])
  const [apiKey, setApiKey] = useState('')
  const [text, setText] = useState('')
  const [result, setResult] = useState<Translation | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchLanguages().then((data) => {
      setLanguages(data.languages)
      if (data.languages.length > 0) setLangCode(data.languages[0].code)
    })
  }, [])

  const selectedProvider = PROVIDERS.find((p) => p.value === provider)!

  // Update model when provider changes
  useEffect(() => {
    setModel(selectedProvider.models[0])
  }, [provider])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!text.trim() || !langCode) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const translation = await translatePipeline(
        {
          text: text.trim(),
          language_code: langCode,
          agent: { provider, model },
        },
        apiKey || undefined,
      )
      setResult(translation)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Translation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <h1 className="mb-2 text-2xl font-bold">Translate</h1>
      <p className="mb-8 text-sm text-gray-500">
        Translate English text using the pipeline translator with structured sentence types.
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Text input */}
        <div>
          <label className="mb-1 block text-sm font-medium text-gray-700">English text</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none"
            placeholder="Enter English text to translate..."
          />
        </div>

        {/* Language + Provider + Model */}
        <div className="grid gap-4 sm:grid-cols-3">
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">Language</label>
            <select
              value={langCode}
              onChange={(e) => setLangCode(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none"
            >
              {languages.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name} ({l.code})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">Provider</label>
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none"
            >
              {PROVIDERS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none"
            >
              {selectedProvider.models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* API key */}
        {provider !== 'ollama' && (
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">
              API Key{' '}
              <span className="font-normal text-gray-400">(or set env var on server)</span>
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none"
              placeholder="sk-..."
            />
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !text.trim()}
          className="rounded-lg bg-indigo-600 px-6 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? 'Translating...' : 'Translate'}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="mt-8 space-y-4">
          <h2 className="text-lg font-semibold">Translation Result</h2>

          <div className="rounded-xl border border-gray-200 bg-white shadow-sm">
            <div className="grid gap-px bg-gray-100 md:grid-cols-2">
              <div className="bg-white p-4">
                <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-gray-400">
                  English
                </p>
                <p className="text-sm">{result.english}</p>
              </div>
              <div className="bg-white p-4">
                <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-gray-400">
                  Translation
                </p>
                <p className="font-mono text-sm text-indigo-700">{result.translation}</p>
              </div>
            </div>
          </div>

          {/* Back-translations */}
          {result.back_translations.length > 0 && (
            <div>
              <h3 className="mb-2 text-sm font-semibold text-gray-600">Back-translations</h3>
              <div className="space-y-1">
                {result.back_translations.map((bt, i) => (
                  <div key={i} className="rounded-lg bg-gray-50 px-3 py-2 text-sm">
                    <span className="font-mono text-xs text-gray-400">{bt.sentence_type}</span>
                    <span className="mx-2 text-gray-300">|</span>
                    <span className="text-gray-700">{bt.english}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Structured sentences */}
          <details className="rounded-lg border border-gray-200 bg-white">
            <summary className="cursor-pointer px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900">
              Structured data ({result.sentences.length} sentence{result.sentences.length !== 1 && 's'})
            </summary>
            <pre className="max-h-80 overflow-auto border-t border-gray-100 p-4 text-xs text-gray-700">
              {JSON.stringify(result.sentences, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  )
}
