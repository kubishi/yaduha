import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { fetchLanguage } from '../api/languages'
import { fetchSchema } from '../api/schemas'
import type { LanguageDetail, SentenceSchemaResponse } from '../types/api'
import SchemaGraph from '../components/graph/SchemaGraph'

export default function LanguagePage() {
  const { code } = useParams<{ code: string }>()
  const [language, setLanguage] = useState<LanguageDetail | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [schema, setSchema] = useState<SentenceSchemaResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!code) return
    fetchLanguage(code)
      .then((data) => {
        setLanguage(data)
        if (data.sentence_types.length > 0) {
          setSelected(data.sentence_types[0].name)
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [code])

  useEffect(() => {
    if (!code || !selected) return
    setSchema(null)
    fetchSchema(code, selected).then(setSchema).catch(console.error)
  }, [code, selected])

  if (loading) {
    return <div className="flex min-h-[60vh] items-center justify-center">Loading...</div>
  }
  if (error || !language) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center text-red-600">
        {error ?? 'Language not found'}
      </div>
    )
  }

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)]">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 border-r border-gray-200 bg-white">
        <div className="border-b border-gray-200 p-4">
          <h2 className="text-lg font-semibold">{language.name}</h2>
          <span className="font-mono text-xs text-gray-400">{language.code}</span>
        </div>
        <nav className="p-2">
          <p className="px-2 py-1 text-xs font-semibold uppercase tracking-wider text-gray-400">
            Sentence Types
          </p>
          {language.sentence_types.map((st) => (
            <button
              key={st.name}
              onClick={() => setSelected(st.name)}
              className={`w-full rounded-md px-3 py-2 text-left text-sm transition-colors ${
                selected === st.name
                  ? 'bg-indigo-50 font-medium text-indigo-700'
                  : 'text-gray-700 hover:bg-gray-50'
              }`}
            >
              <div>{st.name}</div>
              <div className="text-xs text-gray-400">{st.field_count} fields</div>
            </button>
          ))}
          {selected && (
            <Link
              to={`/languages/${code}/examples/${selected}`}
              className="mt-2 block rounded-md px-3 py-2 text-center text-sm font-medium text-indigo-600 hover:bg-indigo-50"
            >
              View Examples
            </Link>
          )}
        </nav>
      </aside>

      {/* Graph area */}
      <div className="flex-1">
        {schema ? (
          <SchemaGraph schema={schema.json_schema} title={schema.sentence_type} />
        ) : (
          <div className="flex h-full items-center justify-center text-gray-400">
            Loading schema...
          </div>
        )}
      </div>
    </div>
  )
}
