import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHeartPulse, faLungs, faDroplet, faDna, faMars, faVenus } from '@fortawesome/free-solid-svg-icons';
import Navbar from '../components/layout/Navbar';
import Footer from '../components/layout/Footer';
import Button from '../components/common/Button';
import '../styles/global.css';
import heroImage from '../assets/images/hero.png';

const Home: React.FC = () => {
    // Determine auth state (this logic might need to be lifted or context used)
    const isAuthenticated = !!sessionStorage.getItem('user_id');
    const navigate = useNavigate();

    const handleCardClick = (path: string) => {
        navigate(path);
    };

    return (
        <div className="page-wrapper">
            <Navbar isAuthenticated={isAuthenticated} onLogout={() => sessionStorage.removeItem('user_id')} />

            <main style={{ flex: 1 }}>
                {/* Hero Section */}
                <section className="hero-section">
                    <div className="container hero-container">
                        <div className="hero-content">
                            <h1 className="hero-title">
                                Advanced AI for <span className="text-gradient">Early Disease Detection</span>
                            </h1>
                            <p className="hero-subtitle">
                                SAM AI leverages cutting-edge machine learning algorithms to assist in the early diagnosis of critical health conditions with high precision.
                            </p>
                            <div className="hero-actions">
                                <Button size="lg" variant="primary" onClick={() => document.getElementById('diagnostic-modules')?.scrollIntoView({ behavior: 'smooth' })}>
                                    Start Diagnostics
                                </Button>
                                <Link to="/about">
                                    <Button size="lg" variant="secondary">Learn More</Button>
                                </Link>
                            </div>
                        </div>
                        <div className="hero-image-wrapper">
                            <div className="hero-glow"></div>
                            <img src={heroImage} alt="Medical AI Interface" className="hero-image" />
                        </div>
                    </div>
                </section>

                {/* Features Section */}
                <section className="features-section" id="diagnostic-modules">
                    <div className="container">
                        <h2 className="section-title">Diagnostic Modules</h2>
                        <div className="grid-responsive">
                            {/* Heart Disease Card */}
                            <div className="feature-card heart-card" onClick={() => handleCardClick('/heartDisease')}>
                                <div className="card-bg-glow"></div>
                                <div className="feature-icon-wrapper">
                                    <FontAwesomeIcon icon={faHeartPulse} className="feature-icon" />
                                </div>
                                <h3>Heart Disease</h3>
                                <p>Analyze cardiovascular metrics to predict heart disease risks with high accuracy.</p>
                                <Button size="sm" variant="primary" className="width-full mt-auto">Start Analysis</Button>
                            </div>

                            {/* Liver Health Card */}
                            <div className="feature-card liver-card" onClick={() => handleCardClick('/liverDetection')}>
                                <div className="card-bg-glow"></div>
                                <div className="feature-icon-wrapper">
                                    <FontAwesomeIcon icon={faLungs} className="feature-icon" />
                                </div>
                                <h3>Liver Health</h3>
                                <p>Comprehensive analysis of liver function markers for early detection of hepatic disorders.</p>
                                <Button size="sm" variant="primary" className="width-full mt-auto">Start Analysis</Button>
                            </div>

                            {/* Woman Diabetes Card */}
                            <div className="feature-card diabetes-f-card" onClick={() => handleCardClick('/womanDiabetes')}>
                                <div className="card-bg-glow"></div>
                                <div className="feature-icon-wrapper">
                                    <FontAwesomeIcon icon={faVenus} className="feature-icon" style={{ marginRight: '8px' }} />
                                    <FontAwesomeIcon icon={faDroplet} className="feature-icon" />
                                </div>
                                <h3>Diabetes (Female)</h3>
                                <p>Specialized predictive model for diabetes risk assessment in women.</p>
                                <Button size="sm" variant="primary" className="width-full mt-auto">Start Analysis</Button>
                            </div>

                            {/* Man Diabetes Card */}
                            <div className="feature-card diabetes-m-card" onClick={() => handleCardClick('/manDiabetes')}>
                                <div className="card-bg-glow"></div>
                                <div className="feature-icon-wrapper">
                                    <FontAwesomeIcon icon={faMars} className="feature-icon" style={{ marginRight: '8px' }} />
                                    <FontAwesomeIcon icon={faDroplet} className="feature-icon" />
                                </div>
                                <h3>Diabetes (Male)</h3>
                                <p>Specialized predictive model for diabetes risk assessment in men.</p>
                                <Button size="sm" variant="primary" className="width-full mt-auto">Start Analysis</Button>
                            </div>

                            {/* Breast Cancer Card */}
                            <div className="feature-card cancer-card" onClick={() => handleCardClick('/breastCancerDetection')}>
                                <div className="card-bg-glow"></div>
                                <div className="feature-icon-wrapper">
                                    <FontAwesomeIcon icon={faDna} className="feature-icon" />
                                </div>
                                <h3>Breast Cancer</h3>
                                <p>Advanced screening analysis using machine learning on cytology data.</p>
                                <Button size="sm" variant="primary" className="width-full mt-auto">Start Analysis</Button>
                            </div>
                        </div>
                    </div>
                </section>
            </main>

            <Footer />

            <style>{`
                .page-wrapper {
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                }
                .hero-section {
                    padding: 4rem 0;
                    position: relative;
                    overflow: hidden;
                }
                .hero-container {
                    display: flex;
                    align-items: center;
                    gap: 4rem;
                }
                .hero-content {
                    flex: 1;
                    z-index: 10;
                }
                .hero-title {
                    font-size: 3.5rem;
                    margin-bottom: 1.5rem;
                    line-height: 1.1;
                }
                .hero-subtitle {
                    font-size: 1.25rem;
                    margin-bottom: 2.5rem;
                    max-width: 600px;
                }
                .hero-actions {
                    display: flex;
                    gap: 1rem;
                }
                .hero-image-wrapper {
                    flex: 1;
                    position: relative;
                    display: flex;
                    justify-content: center;
                }
                .hero-image {
                    width: 100%;
                    max-width: 600px;
                    border-radius: var(--radius-lg);
                    box-shadow: var(--shadow-glow);
                    border: 1px solid rgba(59, 130, 246, 0.3);
                }
                .hero-glow {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 120%;
                    height: 120%;
                    background: radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, rgba(0,0,0,0) 70%);
                    z-index: -1;
                    filter: blur(40px);
                }
                .features-section {
                    padding: 4rem 0;
                    background: rgba(0,0,0,0.2);
                }
                .section-title {
                    text-align: center;
                    font-size: 2.5rem;
                    margin-bottom: 3rem;
                }
                .grid-responsive {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 2rem;
                }
                
                /* Creative Card Styles */
                .feature-card {
                    position: relative;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                    width: 300px; /* Fixed width */
                    height: 360px; /* Reduced fixed height */
                    padding: 1.75rem; /* Slightly reduced padding */
                    background: rgba(25, 32, 41, 0.6);
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: var(--radius-xl);
                    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                    cursor: pointer;
                    overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    /* No flex: 1 1 ... to prevent resizing */
                    flex-shrink: 0; 
                }

                .card-bg-glow {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: radial-gradient(circle at 50% 0%, rgba(255, 255, 255, 0.05), transparent 70%);
                    opacity: 0;
                    transition: opacity 0.4s ease;
                    z-index: 0;
                }

                .feature-card:hover {
                    transform: translateY(-8px);
                    background: rgba(30, 41, 59, 0.8);
                    border-color: rgba(255, 255, 255, 0.2);
                }

                .feature-card:hover .card-bg-glow {
                    opacity: 1;
                }

                .feature-icon-wrapper {
                    position: relative;
                    z-index: 1;
                    width: 80px;
                    height: 80px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 50%;
                    margin-bottom: 1.5rem;
                    transition: all 0.3s ease;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }

                .feature-icon {
                    font-size: 2.5rem;
                    color: var(--text-secondary);
                    transition: all 0.3s ease;
                }

                .feature-card h3 {
                    position: relative;
                    z-index: 1;
                    margin-bottom: 1rem;
                    font-size: 1.25rem;
                    transition: color 0.3s ease;
                }

                .feature-card p {
                    position: relative;
                    z-index: 1;
                    font-size: 0.95rem;
                    color: var(--text-muted);
                    margin-bottom: 1.5rem;
                    line-height: 1.6;
                }

                .width-full {
                    position: relative;
                    z-index: 1;
                    width: 100%;
                }

                .mt-auto {
                    margin-top: auto;
                }

                /* Individual Card Accents */
                
                /* Heart - Red */
                .heart-card:hover { box-shadow: 0 10px 30px -10px rgba(239, 68, 68, 0.3); border-color: rgba(239, 68, 68, 0.3); }
                .heart-card:hover .feature-icon-wrapper { background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.5); transform: scale(1.1); }
                .heart-card:hover .feature-icon { color: #EF4444; }
                .heart-card:hover h3 { color: #EF4444; }

                /* Liver - Amber/Orange */
                .liver-card:hover { box-shadow: 0 10px 30px -10px rgba(245, 158, 11, 0.3); border-color: rgba(245, 158, 11, 0.3); }
                .liver-card:hover .feature-icon-wrapper { background: rgba(245, 158, 11, 0.1); border-color: rgba(245, 158, 11, 0.5); transform: scale(1.1); }
                .liver-card:hover .feature-icon { color: #F59E0B; }
                .liver-card:hover h3 { color: #F59E0B; }

                /* Diabetes Female - Pink/Purple */
                .diabetes-f-card:hover { box-shadow: 0 10px 30px -10px rgba(236, 72, 153, 0.3); border-color: rgba(236, 72, 153, 0.3); }
                .diabetes-f-card:hover .feature-icon-wrapper { background: rgba(236, 72, 153, 0.1); border-color: rgba(236, 72, 153, 0.5); transform: scale(1.1); }
                .diabetes-f-card:hover .feature-icon { color: #EC4899; }
                .diabetes-f-card:hover h3 { color: #EC4899; }

                /* Diabetes Male - Blue */
                .diabetes-m-card:hover { box-shadow: 0 10px 30px -10px rgba(59, 130, 246, 0.3); border-color: rgba(59, 130, 246, 0.3); }
                .diabetes-m-card:hover .feature-icon-wrapper { background: rgba(59, 130, 246, 0.1); border-color: rgba(59, 130, 246, 0.5); transform: scale(1.1); }
                .diabetes-m-card:hover .feature-icon { color: #3B82F6; }
                .diabetes-m-card:hover h3 { color: #3B82F6; }

                /* Cancer - Rose/Red (more intense pink) */
                .cancer-card:hover { box-shadow: 0 10px 30px -10px rgba(244, 63, 94, 0.3); border-color: rgba(244, 63, 94, 0.3); }
                .cancer-card:hover .feature-icon-wrapper { background: rgba(244, 63, 94, 0.1); border-color: rgba(244, 63, 94, 0.5); transform: scale(1.1); }
                .cancer-card:hover .feature-icon { color: #F43F5E; }
                .cancer-card:hover h3 { color: #F43F5E; }
                
                @media (max-width: 968px) {
                    .hero-container {
                        flex-direction: column;
                        text-align: center;
                        gap: 2rem;
                    }
                    .hero-title {
                        font-size: 2.5rem;
                    }
                    .hero-actions {
                        justify-content: center;
                    }
                    .hero-subtitle {
                        margin: 0 auto 2rem auto;
                    }
                }
            `}</style>
        </div>
    );
};

export default Home;
