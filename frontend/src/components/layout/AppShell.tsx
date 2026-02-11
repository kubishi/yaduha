import type { ReactNode } from 'react'
import NavBar from './NavBar'

export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <NavBar />
      <main className="flex-1">{children}</main>
    </div>
  )
}
