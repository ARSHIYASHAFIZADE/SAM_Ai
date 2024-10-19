import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import Home from './Home';
import Contact from './Contact';
import About from './About';
import ManDiabetes from './ManDiabetes';
import WomanDiabetes from './WomanDiabetes';
import HeartDisease from './HeartDisease';
import LoginPage from "./LoginPage";
import NotFound from "./NotFound";
import RegisterPage from "./RegisterPage";
import LiverDetection from "./LiverDetection";
import BreastCancerDetection from "./BreastCancerDetection";
import BackgroundAnimation from './BackgroundAnimation'; // Import the BackgroundAnimation component

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    useEffect(() => {
        const user = sessionStorage.getItem('user_id');
        if (user) {
            setIsAuthenticated(true);
        }
    }, []);

    const handleLogin = (userId: string) => {
        sessionStorage.setItem('user_id', userId);
        setIsAuthenticated(true);
    };

    const handleLogout = () => {
        sessionStorage.removeItem('user_id');
        setIsAuthenticated(false);
    };

    return (
        <Router>
            <div className="App">
                {/* Background Animation */}
                <BackgroundAnimation />
                <header className="App-header">
                    <nav className="navbar">
                        <div className="navbar-brand">SAM AI</div>
                        <ul className="navbar-nav">
                            <li className="nav-item"><Link to="/">Home</Link></li>
                            {!isAuthenticated && <li className="nav-item"><Link to="/login">Login</Link></li>}
                            {!isAuthenticated && <li className="nav-item"><Link to="/register">Register</Link></li>}
                            {isAuthenticated && <li className="nav-item"><Link to="/" onClick={handleLogout}>Logout</Link></li>}
                            <li className="nav-item"><Link to="/about">About</Link></li>
                            <li className="nav-item"><Link to="/contact">Contact</Link></li>
                        </ul>
                    </nav>
                </header>
                <main>
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/about" element={<About />} />
                        <Route path="/contact" element={<Contact />} />
                        <Route path="/manDiabetes" element={isAuthenticated ? <ManDiabetes /> : <Navigate to="/login" />} />
                        <Route path="/womanDiabetes" element={isAuthenticated ? <WomanDiabetes /> : <Navigate to="/login" />} />
                        <Route path="/heartDisease" element={isAuthenticated ? <HeartDisease /> : <Navigate to="/login" />} />
                        <Route path="/liverDetection" element={isAuthenticated ? <LiverDetection /> : <Navigate to="/login" />} />
                        <Route path="/breastCancerDetection" element={isAuthenticated ? <BreastCancerDetection /> : <Navigate to="/login" />} />
                        <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />
                        <Route path="/register" element={<RegisterPage />} />
                        <Route path="*" element={<NotFound />} />
                    </Routes>
                </main>
                <footer>
                    <p>&copy; 2024 SAM AI. All rights reserved.</p>
                </footer>
            </div>
        </Router>
    );
}

export default App;
