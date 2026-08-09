import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import ForYou from './pages/ForYou';
import Login from './pages/Login';
import Preferences from './pages/Preferences';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/foryou" element={<ForYou />} />
            <Route path="/login" element={<Login />} />
            <Route path="/preferences" element={<Preferences />} />
          </Routes>
        </main>
        <footer className="bg-white border-t border-gray-200 py-8 mt-12 text-center text-gray-500 text-sm">
          <p>© 2026 Jagrit News. All rights reserved.</p>
          <p className="text-xs mt-1">Powered by Microsoft MIND Dataset and XGBoost.</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
