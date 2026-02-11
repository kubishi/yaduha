import { Routes, Route } from 'react-router-dom'
import AppShell from './components/layout/AppShell'
import HomePage from './pages/HomePage'
import LanguagePage from './pages/LanguagePage'
import ExamplesPage from './pages/ExamplesPage'
import TranslatePage from './pages/TranslatePage'

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/languages/:code" element={<LanguagePage />} />
        <Route path="/languages/:code/examples/:sentenceType" element={<ExamplesPage />} />
        <Route path="/translate" element={<TranslatePage />} />
      </Routes>
    </AppShell>
  )
}
