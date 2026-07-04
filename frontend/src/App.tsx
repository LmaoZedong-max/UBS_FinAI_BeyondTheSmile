import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Terminal from './pages/Terminal'
import Alerts from './pages/Alerts'
import Chat from './pages/Chat'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/terminal" replace />} />
        <Route path="/terminal" element={<Terminal />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="*" element={<Navigate to="/terminal" replace />} />
      </Route>
    </Routes>
  )
}
