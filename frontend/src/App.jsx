// frontend/src/App.jsx
import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import UploadForm from "./components/UploadForm";
import ScanStatus from "./components/ScanStatus";
import TestResults from "./components/TestResults";
import RLCharts from "./components/RLCharts";

const Home = ({ onJobCreated }) => (
  <div className="space-y-6">
    <UploadForm onJobCreated={onJobCreated} />
  </div>
);

const Dashboard = ({ jobId }) => (
  <div className="space-y-6">
    <ScanStatus jobId={jobId} />
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <TestResults jobId={jobId} />
      <RLCharts jobId={jobId} />
    </div>
  </div>
);

export default function App() {
  const [activeJobId, setActiveJobId] = useState(null);

  return (
    <Router>
      <div className="min-h-screen bg-gray-950 text-gray-100">
        <header className="bg-gray-900 border-b border-gray-800">
          <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
            <Link to="/" className="text-2xl font-bold">Hybrid Agentic API Tester</Link>
            <nav className="space-x-4 text-sm">
              <Link to="/" className="px-3 py-1 rounded-md bg-gray-800 hover:bg-gray-700">Home</Link>
              <Link to="/dashboard" className="px-3 py-1 rounded-md bg-gray-800 hover:bg-gray-700">Dashboard</Link>
            </nav>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Home onJobCreated={(jid) => setActiveJobId(jid)} />} />
            <Route path="/dashboard" element={<Dashboard jobId={activeJobId} />} />
            <Route path="*" element={
              <div className="text-center p-8">
                <h2 className="text-xl font-semibold">Page not found</h2>
                <p className="text-sm text-gray-400 mt-2">Try the <Link to="/" className="underline">Home</Link> page.</p>
              </div>
            } />
          </Routes>
        </main>

        <footer className="bg-gray-900 border-t border-gray-800 mt-8">
          <div className="max-w-6xl mx-auto px-4 py-4 text-sm text-gray-500">
            © {new Date().getFullYear()} Hybrid Agentic API Tester — Local Demo
          </div>
        </footer>
      </div>
    </Router>
  );
}
