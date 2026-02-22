import React from 'react';
import MainLayout from '../components/layout/MainLayout';
import Card from '../components/common/Card';
import '../styles/global.css';

const AboutPage: React.FC = () => {
    return (
        <MainLayout isAuthenticated={false} onLogout={() => { }}>
            <div className="about-page">
                {/* Hero Section */}
                <section className="about-hero">
                    <div className="hero-content">
                        <h1>Revolutionizing <span className="text-gradient">Healthcare</span></h1>
                        <p className="hero-subtitle">
                            SAM AI combines advanced machine learning with medical expertise to provide
                            early, accurate, and accessible disease detection for everyone.
                        </p>
                    </div>
                </section>

                {/* Mission & Vision */}
                <section className="mission-section">
                    <div className="grid-container">
                        <Card className="mission-card">
                            <h3>Our Mission</h3>
                            <p>
                                To democratize access to high-quality healthcare diagnostics by leveraging
                                artificial intelligence. We believe that early detection is key to saving lives,
                                and technology bridges the gap between symptoms and diagnosis.
                            </p>
                        </Card>
                        <Card className="mission-card">
                            <h3>Our Vision</h3>
                            <p>
                                A world where preventable diseases are caught in their earliest stages,
                                reducing treatment costs and improving patient outcomes globally through
                                intelligent, data-driven insights.
                            </p>
                        </Card>
                    </div>
                </section>

                {/* Technology Section */}
                <section className="tech-section">
                    <div className="tech-content">
                        <h2>Powered by <span className="text-highlight">State-of-the-Art AI</span></h2>
                        <p>
                            Our platform utilizes cutting-edge algorithms including Gradient Boosting Classifiers,
                            Neural Networks, and Logistic Regression models. Trained on thousands of validated
                            medical datasets, SAM AI delivers probability-based risk assessments you can trust.
                        </p>
                        <div className="tech-stats">
                            <div className="stat-item">
                                <span className="stat-number">95%+</span>
                                <span className="stat-label">Accuracy Rate</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-number">5+</span>
                                <span className="stat-label">Disease Models</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-number">24/7</span>
                                <span className="stat-label">Availability</span>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Team Section (Placeholder) */}
                <section className="team-section">
                    <h2>Meet the <span className="text-gradient">Team</span></h2>
                    <div className="team-grid">
                        <Card className="team-card">
                            <div className="avatar-placeholder">AS</div>
                            <h4>Arshiya Shafizade</h4>
                            <span className="role">Lead Developer & Researcher</span>
                            <p>Driving the technical vision and AI model development.</p>
                        </Card>
                        {/* Add more team members here */}
                    </div>
                </section>
            </div>

            <style>{`
                .about-page {
                    padding-bottom: 4rem;
                }
                .about-hero {
                    text-align: center;
                    padding: 6rem 1rem 4rem;
                    background: radial-gradient(circle at center, rgba(59, 130, 246, 0.1), transparent 70%);
                }
                .hero-content h1 {
                    font-size: 3.5rem;
                    margin-bottom: 1rem;
                    line-height: 1.2;
                }
                .hero-subtitle {
                    font-size: 1.2rem;
                    color: var(--text-muted);
                    max-width: 700px;
                    margin: 0 auto;
                }
                
                .mission-section, .tech-section, .team-section {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 4rem 1rem;
                }
                
                .grid-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 2rem;
                }

                .mission-card {
                    padding: 2.5rem;
                    border: 1px solid rgba(59, 130, 246, 0.2);
                }
                .mission-card h3 {
                    font-size: 1.5rem;
                    margin-bottom: 1rem;
                    color: var(--primary);
                }
                .mission-card p {
                    color: var(--text-muted);
                    line-height: 1.6;
                }

                .tech-section {
                    background: rgba(30, 41, 59, 0.3);
                    border-radius: var(--radius-xl);
                    margin: 4rem auto;
                    text-align: center;
                }
                .tech-content h2 {
                    font-size: 2.5rem;
                    margin-bottom: 1.5rem;
                }
                .tech-content p {
                    max-width: 800px;
                    margin: 0 auto 3rem;
                    color: var(--text-muted);
                    font-size: 1.1rem;
                }
                .tech-stats {
                    display: flex;
                    justify-content: center;
                    gap: 4rem;
                    flex-wrap: wrap;
                }
                .stat-item {
                    display: flex;
                    flex-direction: column;
                }
                .stat-number {
                    font-size: 3rem;
                    font-weight: 800;
                    color: var(--secondary);
                }
                .stat-label {
                    color: var(--text-muted);
                    font-size: 0.9rem;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }

                .team-section {
                    text-align: center;
                }
                .team-section h2 {
                    font-size: 2.5rem;
                    margin-bottom: 3rem;
                }
                .team-grid {
                    display: flex;
                    justify-content: center;
                    gap: 2rem;
                    flex-wrap: wrap;
                }
                .team-card {
                    width: 300px;
                    padding: 2rem;
                    text-align: center;
                }
                .avatar-placeholder {
                    width: 80px;
                    height: 80px;
                    background: linear-gradient(135deg, var(--primary), var(--secondary));
                    border-radius: 50%;
                    margin: 0 auto 1.5rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: white;
                }
                .team-card h4 {
                    font-size: 1.25rem;
                    margin-bottom: 0.5rem;
                }
                .role {
                    display: block;
                    color: var(--primary);
                    font-size: 0.9rem;
                    margin-bottom: 1rem;
                    font-weight: 600;
                }
                .team-card p {
                    font-size: 0.9rem;
                    color: var(--text-muted);
                }

                .text-highlight {
                    color: var(--secondary);
                }
            `}</style>
        </MainLayout>
    );
};

export default AboutPage;
