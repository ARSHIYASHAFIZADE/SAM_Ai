import React, { useState } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import Card from '../components/common/Card';
import authImage from '../assets/images/auth_side.png';
import { API_BASE_URL } from '../utils/api';
import '../styles/global.css';

interface LoginPageProps {
    onLogin?: () => void;
}


const LoginPage: React.FC<LoginPageProps> = ({ onLogin }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();
    const location = useLocation();

    // Get the state from the location (where we were redirected from)
    // If no state, default to home page
    const from = location.state?.from?.pathname || "/";

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError('');

        try {
            const response = await fetch(`${API_BASE_URL}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            const data = await response.json();

            if (response.ok) {
                if (onLogin) onLogin();
                navigate(from, { replace: true });

            } else {
                setError(data.error || 'Login failed. Please check your credentials.');
            }
        } catch (err) {
            setError('An error occurred. Please try again later.');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="auth-page">
            <div className="auth-container">
                <div className="auth-form-side">
                    <Card className="auth-card">
                        <h2 className="auth-title">Welcome Back</h2>
                        <p className="auth-subtitle">Sign in to access your medical dashboard</p>

                        <form onSubmit={handleSubmit}>
                            <Input
                                label="Email"
                                type="email"
                                placeholder="name@example.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                            <Input
                                label="Password"
                                type="password"
                                placeholder="Go ahead, enter your password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />

                            {error && <div className="error-message">{error}</div>}

                            <Button
                                type="submit"
                                variant="primary"
                                size="lg"
                                className="w-full"
                                isLoading={isLoading}
                                style={{ width: '100%', marginTop: '1rem' }}
                            >
                                Sign In
                            </Button>
                        </form>

                        <p className="auth-footer">
                            Don't have an account? <Link to="/register">Sign up</Link>
                        </p>
                    </Card>
                </div>
                <div className="auth-image-side">
                    <img src={authImage} alt="Medical AI" className="auth-image" />
                    <div className="auth-overlay">
                        <h3>Secure. Intelligent. Precise.</h3>
                        <p>Join thousands of users utilizing SAM AI for early detection.</p>
                    </div>
                </div>
            </div>

            <style>{`
                .auth-page {
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: var(--background);
                    padding: 1rem;
                }
                .auth-container {
                    display: flex;
                    width: 100%;
                    max-width: 1000px;
                    background: var(--surface);
                    border-radius: var(--radius-xl);
                    overflow: hidden;
                    box-shadow: var(--shadow-2xl);
                    border: 1px solid rgba(255,255,255,0.05);
                    min-height: 600px;
                }
                .auth-form-side {
                    flex: 1;
                    padding: 3rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .auth-image-side {
                    flex: 1;
                    position: relative;
                    display: none;
                    background: radial-gradient(circle at center, #1e293b, #0f172a);
                    overflow: hidden;
                }
                .auth-image {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    mix-blend-mode: multiply;
                    filter: brightness(0.9) contrast(1.1);
                }
                .auth-overlay {
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    padding: 2rem;
                    background: linear-gradient(to top, rgba(15, 23, 42, 0.9), transparent);
                    color: white;
                    z-index: 10;
                }
                .error-message {
                    color: var(--danger);
                    background: rgba(239, 68, 68, 0.1);
                    padding: 0.75rem;
                    border-radius: var(--radius-md);
                    margin-bottom: 1rem;
                    font-size: 0.9rem;
                }
                
                @media (min-width: 768px) {
                    .auth-image-side {
                        display: block;
                    }
                }
            `}</style>
        </div>
    );
};

export default LoginPage;
