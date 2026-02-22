import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHeartPulse, faStethoscope, faChartLine, faUserMd, faInfoCircle } from '@fortawesome/free-solid-svg-icons';
import MainLayout from '../components/layout/MainLayout';
import Card from '../components/common/Card';
import Input from '../components/common/Input';
import Select from '../components/common/Select';
import Button from '../components/common/Button';
import Toast from '../components/common/Toast';
import { API_BASE_URL } from '../utils/api';
import '../styles/global.css';

interface PredictionResult {
    prediction: number;
    probability: string;
}

const INITIAL_STATE = {
    age: '', sex: '', cp: '', trestbps: '',
    chol: '', fbs: '', restecg: '', thalach: '',
    exang: '', oldpeak: '', slope: '', ca: '', thal: ''
};

const HeartDisease: React.FC = () => {
    const [inputData, setInputData] = useState(INITIAL_STATE);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

    // Refs
    const infoRef = useRef<HTMLDivElement>(null);
    const resultRef = useRef<HTMLDivElement>(null);

    // Scroll Animation Observer
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                    }
                });
            },
            { threshold: 0.1 }
        );

        if (infoRef.current) {
            const sections = infoRef.current.querySelectorAll('.animate-on-scroll');
            sections.forEach((section) => observer.observe(section));
        }

        return () => observer.disconnect();
    }, [result]);

    // Scroll to result
    useEffect(() => {
        if (result && resultRef.current) {
            resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, [result]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setInputData({ ...inputData, [e.target.name]: e.target.value });
    };

    const handleReset = () => {
        setResult(null);
        setInputData(INITIAL_STATE);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const validateForm = () => {
        if (!inputData.age || !inputData.trestbps || !inputData.chol) {
            setToast({ message: "Please fill all required fields", type: 'error' });
            return false;
        }
        return true;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!validateForm()) return;
        setIsLoading(true);

        const formattedData = {
            ...inputData,
            age: Number(inputData.age),
            trestbps: Number(inputData.trestbps),
            chol: Number(inputData.chol),
            thalach: Number(inputData.thalach),
            oldpeak: Number(inputData.oldpeak),
            ca: Number(inputData.ca)
        };

        try {
            const response = await axios.post(`${API_BASE_URL}/detect_heart`, { data: formattedData });
            setResult(response.data);
            setToast({ message: "Analysis complete.", type: 'success' });
        } catch (error) {
            console.error('Error:', error);
            setToast({ message: "Analysis failed. Please try again.", type: 'error' });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <MainLayout isAuthenticated={true} onLogout={() => sessionStorage.removeItem('user_id')}>
            <div className="page-container">
                {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

                {/* Hero / Split Screen Section */}
                <div className="hero-wrapper">
                    <div className="hero-split-container">
                        {/* LEFT: Visuals & Title */}
                        <div className="hero-visuals fade-in-right">
                            <div className="visual-content">
                                <div className="pulse-circle-container">
                                    <div className="pulse-circle"></div>
                                    <div className="icon-wrapper">
                                        <FontAwesomeIcon icon={faHeartPulse} className="main-icon" />
                                    </div>
                                </div>
                                <h1 className="page-title">Heart <br /><span className="text-gradient">Health</span></h1>
                                <p className="page-subtitle">Advanced cardiovascular risk assessment powered by AI.</p>
                                <div className="stats-row">
                                    <div className="stat-item">
                                        <FontAwesomeIcon icon={faStethoscope} className="stat-icon" />
                                        <span>Clinical Vitals</span>
                                    </div>
                                    <div className="stat-item">
                                        <FontAwesomeIcon icon={faChartLine} className="stat-icon" />
                                        <span>Risk Analysis</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* RIGHT: Form / Result */}
                        <div className="hero-form fade-in-left">
                            {!result ? (
                                <Card className="fixed-height-card">
                                    <div className="scrollable-form-box">
                                        <form onSubmit={handleSubmit} className="compact-grid-form four-col">
                                            {/* Row 1 */}
                                            <div className="compact-input">
                                                <Input label="Age" name="age" type="number" value={inputData.age} onChange={handleChange} required placeholder="Years" />
                                            </div>
                                            <div className="compact-input">
                                                <Select label="Sex" name="sex" value={inputData.sex} onChange={handleChange} options={[
                                                    { value: "1", label: "Male" },
                                                    { value: "0", label: "Female" }
                                                ]} required />
                                            </div>
                                            <div className="compact-input">
                                                <Select label="Chest Pain" name="cp" value={inputData.cp} onChange={handleChange} options={[
                                                    { value: "0", label: "Typical Angina" },
                                                    { value: "1", label: "Atypical Angina" },
                                                    { value: "2", label: "Non-anginal" },
                                                    { value: "3", label: "Asymptomatic" }
                                                ]} required />
                                            </div>
                                            <div className="compact-input">
                                                <Input label="Resting BP" name="trestbps" type="number" value={inputData.trestbps} onChange={handleChange} required placeholder="mm Hg" />
                                            </div>

                                            {/* Row 2 */}
                                            <div className="compact-input">
                                                <Input label="Cholesterol" name="chol" type="number" value={inputData.chol} onChange={handleChange} required placeholder="mg/dl" />
                                            </div>
                                            <div className="compact-input">
                                                <Select label="Fasting BS > 120" name="fbs" value={inputData.fbs} onChange={handleChange} options={[
                                                    { value: "1", label: "True" },
                                                    { value: "0", label: "False" }
                                                ]} required />
                                            </div>
                                            <div className="compact-input">
                                                <Select label="Resting ECG" name="restecg" value={inputData.restecg} onChange={handleChange} options={[
                                                    { value: "0", label: "Normal" },
                                                    { value: "1", label: "ST-T Abnormality" },
                                                    { value: "2", label: "LV Hypertrophy" }
                                                ]} required />
                                            </div>
                                            <div className="compact-input">
                                                <Input label="Max Heart Rate" name="thalach" type="number" value={inputData.thalach} onChange={handleChange} required />
                                            </div>

                                            {/* Row 3 */}
                                            <div className="compact-input">
                                                <Select label="Ex. Angina" name="exang" value={inputData.exang} onChange={handleChange} options={[
                                                    { value: "1", label: "Yes" },
                                                    { value: "0", label: "No" }
                                                ]} required />
                                            </div>
                                            <div className="compact-input">
                                                <Input label="ST Depression" name="oldpeak" type="number" step="0.1" value={inputData.oldpeak} onChange={handleChange} required />
                                            </div>
                                            <div className="compact-input">
                                                <Select label="Slope" name="slope" value={inputData.slope} onChange={handleChange} options={[
                                                    { value: "0", label: "Upsloping" },
                                                    { value: "1", label: "Flat" },
                                                    { value: "2", label: "Downsloping" }
                                                ]} required />
                                            </div>
                                            <div className="compact-input">
                                                <Input label="Major Vessels" name="ca" type="number" min="0" max="3" value={inputData.ca} onChange={handleChange} required />
                                            </div>

                                            {/* Row 4 */}
                                            <div className="compact-input" style={{ gridColumn: 'span 2' }}>
                                                <Select label="Thalassemia" name="thal" value={inputData.thal} onChange={handleChange} options={[
                                                    { value: "1", label: "Normal" },
                                                    { value: "2", label: "Fixed Defect" },
                                                    { value: "3", label: "Reversable" }
                                                ]} required />
                                            </div>

                                            <div className="form-actions-fixed full-width-col" style={{ gridColumn: '1 / -1', padding: '0.75rem', marginTop: '0.5rem' }}>
                                                <Button type="submit" variant="primary" size="lg" isLoading={isLoading} className="w-full glow-button compact-btn" style={{ fontSize: '1rem', padding: '0.6rem' }}>
                                                    Assess Heart Health
                                                </Button>
                                            </div>
                                        </form>
                                    </div>
                                </Card>
                            ) : (
                                <Card className="result-card fade-in-up" ref={resultRef}>
                                    <div className="result-header-row">
                                        <div className="result-icon pulse-animation">{result.prediction === 1 ? '⚠' : '✓'}</div>
                                        <div>
                                            <div className="result-title">Analysis Result</div>
                                            <div className="result-value" style={{ color: result.prediction === 1 ? 'var(--danger)' : 'var(--success)' }}>
                                                {result.prediction === 1 ? 'High Risk Detected' : 'Low Risk Profile'}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="result-probability">
                                        Probability: <span className="highlight">{(parseFloat(result.probability) * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="result-guidance compact-guidance">
                                        {result.prediction === 1 ? (
                                            <p>Results suggest a higher likelihood of heart disease. Please consult a cardiologist.</p>
                                        ) : (
                                            <p>Results are within a lower risk category. Maintain a healthy lifestyle.</p>
                                        )}
                                    </div>
                                    <Button onClick={handleReset} variant="secondary" className="mt-4 w-full">New Assessment</Button>
                                </Card>
                            )}
                        </div>
                    </div>
                </div>

                {/* Educational Content - Restored */}
                <div className="info-panel-wrapper" ref={infoRef}>
                    <div className="content-wrapper" style={{ maxWidth: '1000px', width: '100%' }}>

                        <div className="info-grid animate-on-scroll delay-2">
                            <div className="info-card">
                                <h3><FontAwesomeIcon icon={faHeartPulse} /> About Heart Disease</h3>
                                <p>Heart disease describes a range of conditions that affect your heart. Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease; heart rhythm problems (arrhythmias); and heart defects you're born with (congenital heart defects).</p>
                            </div>
                            <div className="info-card">
                                <h3><FontAwesomeIcon icon={faUserMd} /> Risk Factors</h3>
                                <ul>
                                    <li><strong>Age:</strong> Risk increases as you get older.</li>
                                    <li><strong>Sex:</strong> Men are generally at greater risk.</li>
                                    <li><strong>Levels:</strong> High blood pressure and cholesterol.</li>
                                    <li><strong>Lifestyle:</strong> Smoking, diet, and stress.</li>
                                </ul>
                            </div>
                            <div className="info-card">
                                <h3><FontAwesomeIcon icon={faInfoCircle} /> Prevention</h3>
                                <p>You can prevent or treat some forms of heart disease with healthy lifestyle choices. This includes a heart-healthy diet, regular exercise, maintaining a healthy weight, and not smoking.</p>
                            </div>
                        </div>

                        <div className="disclaimer-box animate-on-scroll delay-3">
                            <strong>Medical Disclaimer:</strong> This tool is for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
                        </div>
                    </div>
                </div>
            </div>

            <style>{`
                .page-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    width: 100%;
                    overflow-x: hidden;
                    background: var(--background-dark);
                    min-height: 100vh;
                }

                .hero-wrapper {
                    height: calc(100vh - 80px);
                    max-height: calc(100vh - 80px);
                    width: 100%;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    position: relative;
                    padding: 1rem var(--spacing-lg) 0;
                    overflow: hidden;
                }

                .hero-split-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 1rem;
                    max-width: 1200px;
                    width: 100%;
                    margin: 0 auto;
                    height: 100%;
                    justify-content: center;
                }

                @media (min-width: 1024px) {
                    .hero-split-container {
                        flex-direction: row;
                        flex-direction: row;
                        align-items: flex-start; /* Aligned to top */
                        padding-top: 4rem; /* Added padding to push down slightly */
                        justify-content: space-between;
                        gap: 2rem;
                    }

                    .hero-visuals {
                        flex: 1;
                        max-width: 45%;
                        display: flex;
                        flex-direction: column;
                        /* justify-content: center; Removed to allow top alignment */
                    }

                    .hero-form {
                        flex: 1;
                        max-width: 800px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                }

                .fixed-height-card {
                    padding: 0 !important;
                    background: rgba(30, 41, 59, 0.85) !important;
                    border: 1px solid rgba(255,255,255,0.1) !important;
                    display: flex;
                    flex-direction: column;
                    height: auto;
                    max-height: 650px;
                    width: 100%;
                    overflow: hidden;
                    border-radius: 12px !important;
                }

                .scrollable-form-box {
                    padding: 1.25rem;
                    overflow-y: auto;
                    flex: 1;
                    scrollbar-width: thin;
                    scrollbar-color: var(--primary) rgba(255,255,255,0.1);
                }
                .scrollable-form-box::-webkit-scrollbar {width: 6px; }
                .scrollable-form-box::-webkit-scrollbar-track {background: rgba(255,255,255,0.05); }
                .scrollable-form-box::-webkit-scrollbar-thumb {background-color: var(--primary); border-radius: 10px; }

                /* GRID LAYOUT FIXES */
                .compact-grid-form {
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 0.75rem;
                    width: 100%;
                }

                .compact-grid-form.four-col { grid-template-columns: repeat(4, 1fr); }

                @media (max-width: 1200px) { .compact-grid-form.four-col { grid-template-columns: repeat(3, 1fr); } }
                @media (max-width: 992px) { .compact-grid-form.four-col { grid-template-columns: repeat(2, 1fr); } }
                @media (max-width: 768px) { .compact-grid-form.four-col { grid-template-columns: 1fr; } }
                
                .compact-input {
                    display: flex;
                    flex-direction: column;
                    width: 100%;
                }

                .compact-input > div {
                    margin-bottom: 0 !important;
                    width: 100% !important;
                }

                /* STRICT INPUT STYLING (Override Input.tsx defaults) */
                .compact-input input, .compact-input select {
                    padding: 0.3rem 0.5rem !important;
                    font-size: 0.85rem !important;
                    height: 36px !important; /* Slightly larger for visibility */
                    line-height: normal;
                    width: 100% !important;
                    background: rgba(255, 255, 255, 0.05) !important; /* Increase visibility */
                    border: 1px solid rgba(255, 255, 255, 0.15) !important;
                    color: var(--text-main) !important;
                    border-radius: 6px !important;
                }
                
                .compact-input input:focus, .compact-input select:focus {
                     border-color: var(--primary) !important;
                     box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2) !important;
                     background: rgba(255, 255, 255, 0.1) !important;
                }

                .compact-input label {
                    font-size: 0.75rem !important;
                    margin-bottom: 0.25rem !important;
                    color: var(--text-secondary) !important;
                    display: block !important;
                    font-weight: 500 !important;
                }

                .full-width-col {grid-column: span 4; }
                @media (max-width: 1200px) { .full-width-col {grid-column: span 3; } }
                @media (max-width: 992px) { .full-width-col {grid-column: span 2; } }
                @media (max-width: 768px) { .full-width-col {grid-column: span 1; } }

                .form-actions-fixed {
                    padding: 1rem 1.25rem;
                    background: transparent;
                    /* border-top: 1px solid rgba(255,255,255,0.05); Removed border as per user request */
                }

                /* Educational Content Styles */
                .info-panel-wrapper {
                    padding: 4rem var(--spacing-lg);
                    width: 100%;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    background: rgba(15, 20, 25, 0.5); /* Slight separation */
                }
                
                .info-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 2rem;
                    width: 100%;
                    margin-bottom: 3rem;
                }
                
                .info-card {
                    background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 12px;
                    border: 1px solid rgba(255,255,255,0.05);
                    transition: transform 0.3s ease;
                }
                
                .info-card:hover {
                    transform: translateY(-5px);
                    background: rgba(30, 41, 59, 0.7);
                    border-color: var(--primary-light);
                }
                
                .info-card h3 {
                    color: var(--primary);
                    font-size: 1.2rem;
                    margin-bottom: 1rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                .info-card p, .info-card li {
                    color: var(--text-secondary);
                    font-size: 0.95rem;
                    line-height: 1.6;
                }
                
                .info-card ul {
                    padding-left: 1.25rem;
                    margin: 0;
                }
                
                .info-card li {
                    margin-bottom: 0.5rem;
                }

                 /* Result Card Specific Styles */
                .result-card {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    padding: 2rem;
                    min-height: 400px;
                    background: rgba(30, 41, 59, 0.85) !important;
                    border: 1px solid rgba(255,255,255,0.1) !important;
                    border-radius: 12px !important;
                }

                .result-header-row {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    margin-bottom: 1.5rem;
                }

                .result-icon {
                    font-size: 3rem;
                    line-height: 1;
                    color: var(--primary);
                }

                .result-title {
                    font-size: 1.2rem;
                    color: var(--text-light);
                    margin-bottom: 0.25rem;
                }

                .result-value {
                    font-size: 1.8rem;
                    font-weight: bold;
                }

                .result-probability {
                    font-size: 1.1rem;
                    color: var(--text-main);
                    margin-bottom: 1.5rem;
                }

                .result-probability .highlight {
                    color: var(--primary);
                    font-weight: bold;
                }

                .result-guidance {
                    max-width: 400px;
                    color: var(--text-light);
                    line-height: 1.6;
                    margin-bottom: 2rem;
                }

                .compact-guidance p {
                    margin-bottom: 0;
                }

                /* Visuals */
                .visual-content {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    width: 100%;
                }

                .pulse-circle-container {
                    width: 100px;
                    height: 100px;
                    margin-bottom: 1.5rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    position: relative;
                }
                    justify-content: center;
                    position: relative;
                }

                .pulse-circle {
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    background: radial-gradient(circle, rgba(244, 63, 94, 0.4) 0%, rgba(0,0,0,0) 70%);
                    border-radius: 50%;
                    animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
                }

                .main-icon {
                    font-size: 3.5rem;
                    color: #f43f5e;
                    filter: drop-shadow(0 0 15px rgba(244, 63, 94, 0.6));
                    animation: float-icon 3s ease-in-out infinite;
                }

                .page-title {
                    font-size: 3rem;
                    line-height: 1.1;
                    margin-bottom: 0.5rem;
                    font-weight: 700;
                    text-align: center;
                }

                .text-gradient {
                    background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }

                .page-subtitle {
                    font-size: 1.1rem;
                    color: var(--text-secondary);
                    margin-bottom: 1.5rem;
                    text-align: center;
                    max-width: 400px;
                }

                .stats-row {
                    display: flex;
                    gap: 2rem;
                    margin-top: 1rem;
                }

                .stat-item {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    color: var(--text-muted);
                    font-size: 0.85rem;
                }

                .stat-icon {
                    color: var(--primary);
                }

                /* Animations */
                @keyframes pulse-ring {0% { transform: scale(0.5); opacity: 0; } 50% {opacity: 0.5; } 100% {transform: scale(1.5); opacity: 0; } }
                @keyframes float-icon {0%, 100% { transform: translateY(0); } 50% {transform: translateY(-10px); } }
                .fade-in-right {animation: fadeInRight 0.8s ease-out; }
                .fade-in-left {animation: fadeInLeft 0.8s ease-out; }
                .fade-in-up {animation: fadeInUp 0.8s ease-out; }
                @keyframes fadeInRight {from {opacity: 0; transform: translateX(-30px); } to {opacity: 1; transform: translateX(0); } }
                @keyframes fadeInLeft {from {opacity: 0; transform: translateX(30px); } to {opacity: 1; transform: translateX(0); } }
                @keyframes fadeInUp {from {opacity: 0; transform: translateY(30px); } to {opacity: 1; transform: translateY(0); } }
            `}</style>
        </MainLayout>
    );
};

export default HeartDisease;
