import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Home from './pages/Home';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import HeartDisease from './pages/HeartDisease';
import LiverDetection from './pages/LiverDetection';
import ManDiabetes from './pages/ManDiabetes';
import WomanDiabetes from './pages/WomanDiabetes';
import BreastCancerDetection from './pages/BreastCancerDetection';
import './styles/global.css';

import AboutPage from './pages/AboutPage';

import GooeyBackground from './components/common/GooeyBackground';
import RequireAuth from './components/auth/RequireAuth';
import ScrollToTop from './components/common/ScrollToTop';
import { API_BASE_URL } from './utils/api';

const PlaceholderPage = ({ title }: { title: string }) => (
    <div style={{ padding: '4rem', textAlign: 'center', color: 'white' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>{title}</h1>
        <p>Coming Soon with the new design.</p>
        <a href="/" style={{ color: 'var(--primary)', marginTop: '1rem', display: 'inline-block' }}>Go Home</a>
    </div>
);

const App = () => {
    // Fix 7: Auth state is in-memory only — no sessionStorage.
    // On mount, verify session with backend /@me to handle page refreshes.
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [authChecked, setAuthChecked] = useState(false);

    useEffect(() => {
        fetch(`${API_BASE_URL}/@me`, { credentials: 'include' })
            .then(res => {
                setIsAuthenticated(res.ok);
            })
            .catch(() => setIsAuthenticated(false))
            .finally(() => setAuthChecked(true));
    }, []);

    const handleLogin = () => {
        setIsAuthenticated(true);
    };

    const handleLogout = () => {
        fetch(`${API_BASE_URL}/logout`, { method: 'POST', credentials: 'include' })
            .finally(() => setIsAuthenticated(false));
    };

    // Wait for session check before rendering protected routes to avoid redirect flicker
    if (!authChecked) return null;

    return (
        <Router>
            <ScrollToTop />
            <div className="App">
                <GooeyBackground />
                <Routes>
                    <Route path="/" element={<Home onLogout={handleLogout} isAuthenticated={isAuthenticated} />} />
                    <Route path="/about" element={<AboutPage />} />
                    <Route path="/contact" element={<PlaceholderPage title="Contact Us" />} />

                    {/* Auth Routes */}
                    <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />
                    <Route path="/register" element={<RegisterPage />} />

                    {/* Protected Routes */}
                    <Route path="/heartDisease" element={
                        <RequireAuth isAuthenticated={isAuthenticated}>
                            <HeartDisease onLogout={handleLogout} />
                        </RequireAuth>
                    } />
                    <Route path="/liverDetection" element={
                        <RequireAuth isAuthenticated={isAuthenticated}>
                            <LiverDetection onLogout={handleLogout} />
                        </RequireAuth>
                    } />
                    <Route path="/manDiabetes" element={
                        <RequireAuth isAuthenticated={isAuthenticated}>
                            <ManDiabetes onLogout={handleLogout} />
                        </RequireAuth>
                    } />
                    <Route path="/womanDiabetes" element={
                        <RequireAuth isAuthenticated={isAuthenticated}>
                            <WomanDiabetes onLogout={handleLogout} />
                        </RequireAuth>
                    } />
                    <Route path="/breastCancerDetection" element={
                        <RequireAuth isAuthenticated={isAuthenticated}>
                            <BreastCancerDetection onLogout={handleLogout} />
                        </RequireAuth>
                    } />

                    {/* Fallback */}
                    <Route path="*" element={<Navigate to="/" />} />
                </Routes>
            </div>
        </Router>
    );
}

export default App;
