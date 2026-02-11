import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchLanguages } from '../api/languages'
import type { LanguageSummary } from '../types/api'

export default function HomePage() {
  const [languages, setLanguages] = useState<LanguageSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchLanguages()
      .then((data) => setLanguages(data.languages))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Centered>Loading languages...</Centered>
  if (error) return <Centered className="text-red-600">Error: {error}</Centered>
  if (languages.length === 0) return <Centered>No languages installed.</Centered>

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <h1 className="mb-2 text-2xl font-bold">Languages</h1>
      <p className="mb-8 text-gray-500">
        Explore installed language packages and their sentence structures.
      </p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {languages.map((lang) => (
          <LanguageCard key={lang.code} language={lang} />
        ))}
      </div>
    </div>
  )
}

function LanguageCard({ language }: { language: LanguageSummary }) {
  return (
    <Link
      to={`/languages/${language.code}`}
      className="group rounded-xl border border-gray-200 bg-white p-5 shadow-sm transition-all hover:border-indigo-300 hover:shadow-md"
    >
      <div className="mb-1 flex items-baseline gap-2">
        <span className="text-lg font-semibold text-gray-900 group-hover:text-indigo-600">
          {language.name}
        </span>
        <span className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-xs text-gray-500">
          {language.code}
        </span>
      </div>
      <p className="text-sm text-gray-500">
        {language.sentence_type_count} sentence type{language.sentence_type_count !== 1 && 's'}
      </p>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {language.sentence_types.map((st) => (
          <span
            key={st}
            className="rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700"
          >
            {st}
          </span>
        ))}
      </div>
    </Link>
  )
}

function Centered({
  children,
  className = '',
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={`flex min-h-[60vh] items-center justify-center text-lg ${className}`}>
      {children}
    </div>
  )
}
