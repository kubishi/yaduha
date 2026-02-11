import { Link, useLocation } from 'react-router-dom'

const links = [
  { to: '/', label: 'Languages' },
  { to: '/translate', label: 'Translate' },
]

export default function NavBar() {
  const { pathname } = useLocation()

  return (
    <nav className="border-b border-gray-200 bg-white">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-8 px-4">
        <Link to="/" className="text-lg font-semibold tracking-tight text-indigo-600">
          Yaduha
        </Link>
        <div className="flex gap-1">
          {links.map(({ to, label }) => {
            const active = to === '/' ? pathname === '/' : pathname.startsWith(to)
            return (
              <Link
                key={to}
                to={to}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  active
                    ? 'bg-indigo-50 text-indigo-700'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`}
              >
                {label}
              </Link>
            )
          })}
        </div>
      </div>
    </nav>
  )
}
