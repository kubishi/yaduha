import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { fetchExamples } from '../api/schemas'
import type { ExamplePair } from '../types/api'

export default function ExamplesPage() {
  const { code, sentenceType } = useParams<{ code: string; sentenceType: string }>()
  const [examples, setExamples] = useState<ExamplePair[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!code || !sentenceType) return
    fetchExamples(code, sentenceType)
      .then((data) => setExamples(data.examples))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [code, sentenceType])

  if (loading) {
    return <div className="flex min-h-[60vh] items-center justify-center">Loading examples...</div>
  }
  if (error) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center text-red-600">
        Error: {error}
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <div className="mb-6 flex items-center gap-3">
        <Link
          to={`/languages/${code}`}
          className="text-sm text-indigo-600 hover:text-indigo-800"
        >
          &larr; Back to {code}
        </Link>
      </div>
      <h1 className="mb-2 text-2xl font-bold">{sentenceType} Examples</h1>
      <p className="mb-8 text-sm text-gray-500">
        {examples.length} example{examples.length !== 1 && 's'} showing English input,
        structured representation, and rendered output.
      </p>
      <div className="space-y-4">
        {examples.map((ex, i) => (
          <ExampleCard key={i} example={ex} />
        ))}
      </div>
    </div>
  )
}

function ExampleCard({ example }: { example: ExamplePair }) {
  const [showJson, setShowJson] = useState(false)

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div className="grid gap-px bg-gray-100 md:grid-cols-3">
        {/* English */}
        <div className="bg-white p-4">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-gray-400">
            English
          </p>
          <p className="text-sm text-gray-800">{example.english}</p>
        </div>
        {/* Rendered */}
        <div className="bg-white p-4">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-gray-400">
            Rendered
          </p>
          <p className="font-mono text-sm text-indigo-700">{example.rendered}</p>
        </div>
        {/* Structured */}
        <div className="bg-white p-4">
          <div className="mb-1 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
              Structured
            </p>
            <button
              onClick={() => setShowJson((v) => !v)}
              className="text-xs text-indigo-600 hover:text-indigo-800"
            >
              {showJson ? 'Hide' : 'Show'} JSON
            </button>
          </div>
          {showJson && (
            <pre className="mt-1 max-h-60 overflow-auto rounded bg-gray-50 p-2 text-xs text-gray-700">
              {JSON.stringify(example.structured, null, 2)}
            </pre>
          )}
          {!showJson && (
            <p className="text-xs text-gray-500">
              {Object.keys(example.structured).length} fields
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
