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

// Simple placeholder for Contact to prevent errors if not migrated yet
const PlaceholderPage = ({ title }: { title: string }) => (
    <div style={{ padding: '4rem', textAlign: 'center', color: 'white' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>{title}</h1>
        <p>Coming Soon with the new design.</p>
        <a href="/" style={{ color: 'var(--primary)', marginTop: '1rem', display: 'inline-block' }}>Go Home</a>
    </div>
);

const App = () => {
    // Lazy initialization of auth state to valid initial flicker/redirects
    const [isAuthenticated, setIsAuthenticated] = useState(() => {
        return !!sessionStorage.getItem('user_id');
    });

    useEffect(() => {
        // Double check in case storage changes (optional, but good practice)
        const user = sessionStorage.getItem('user_id');
        if (user) {
            setIsAuthenticated(true);
        }
    }, []);

    const handleLogin = (userId: string) => {
        sessionStorage.setItem('user_id', userId);
        setIsAuthenticated(true);
    };

    return (
        <Router>
            <ScrollToTop />
            <div className="App">
                <GooeyBackground />
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/about" element={<AboutPage />} />
                    <Route path="/contact" element={<PlaceholderPage title="Contact Us" />} />

                    {/* Auth Routes */}
                    <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />
                    <Route path="/register" element={<RegisterPage />} />

                    {/* Protected Routes */}
                    <Route path="/heartDisease" element={
                        <RequireAuth isAuthenticated={isAuthenticated}>
                            <HeartDisease />
                        </RequireAuth>
                    } />
                    <Route path="/liverDetection" element={
                        <RequireAuth isAuthenticated={isAuthenticated}>
                            <LiverDetection />
                        </RequireAuth>
                    } />
                    <Route path="/manDiabetes" element={
                        <RequireAuth isAuthenticated={isAuthenticated}>
                            <ManDiabetes />
                        </RequireAuth>
                    } />
                    <Route path="/womanDiabetes" element={
                        <RequireAuth isAuthenticated={isAuthenticated}>
                            <WomanDiabetes />
                        </RequireAuth>
                    } />
                    <Route path="/breastCancerDetection" element={
                        <RequireAuth isAuthenticated={isAuthenticated}>
                            <BreastCancerDetection />
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
