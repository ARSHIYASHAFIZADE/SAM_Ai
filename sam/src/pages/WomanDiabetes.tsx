import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPersonPregnant, faBaby, faFileMedical, faInfoCircle, faUserMd } from '@fortawesome/free-solid-svg-icons';
import MainLayout from '../components/layout/MainLayout';
import Card from '../components/common/Card';
import Input from '../components/common/Input';
import Button from '../components/common/Button';
import Toast from '../components/common/Toast';
import { API_BASE_URL } from '../utils/api';
import '../styles/global.css';

interface PredictionResult {
    prediction: number;
    probability: number;
}

const INITIAL_STATE = {
    Pregnancies: '', Glucose: '', BloodPressure: '', SkinThickness: '',
    Insulin: '', BMI: '', DiabetesPedigreeFunction: '', Age: ''
};

const WomanDiabetes: React.FC = () => {
    const [inputData, setInputData] = useState(INITIAL_STATE);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

    // Refs for scroll animations
    const infoRef = useRef<HTMLDivElement>(null);
    const resultRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) entry.target.classList.add('visible');
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
        const d = inputData;
        if (!d.Glucose || !d.BloodPressure || !d.BMI) {
            setToast({ message: "Please fill all required fields", type: 'error' });
            return false;
        }
        if (Number(d.Age) < 0 || Number(d.Age) > 120) {
            setToast({ message: "Invalid Age", type: 'error' });
            return false;
        }
        return true;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!validateForm()) return;
        setIsLoading(true);

        const numericData = [
            Number(inputData.Pregnancies),
            Number(inputData.Glucose),
            Number(inputData.BloodPressure),
            Number(inputData.SkinThickness),
            Number(inputData.Insulin),
            Number(inputData.BMI),
            Number(inputData.DiabetesPedigreeFunction),
            Number(inputData.Age)
        ];

        try {
            const response = await axios.post(`${API_BASE_URL}/predict_female`, { input_data: numericData });
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
                                        <FontAwesomeIcon icon={faPersonPregnant} className="main-icon" />
                                    </div>
                                </div>
                                <h1 className="page-title">Diabetes <br /><span className="text-gradient">Risk (Women)</span></h1>
                                <p className="page-subtitle">Specialized assessment for gestational and type 2 diabetes risks</p>
                                <div className="stats-row">
                                    <div className="stat-item">
                                        <FontAwesomeIcon icon={faBaby} className="stat-icon" />
                                        <span>Pregnancy Impact</span>
                                    </div>
                                    <div className="stat-item">
                                        <FontAwesomeIcon icon={faFileMedical} className="stat-icon" />
                                        <span>History Analysis</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* RIGHT: Form / Result */}
                        <div className="hero-form fade-in-left">
                            {!result ? (
                                <Card className="form-card fixed-height-card">
                                    <div className="card-header-area">
                                        <h3 className="form-header">Health Metrics</h3>
                                    </div>
                                    <div className="scrollable-form-box">
                                        <form onSubmit={handleSubmit} className="compact-grid-form four-col">
                                            {/* Row 1 */}
                                            <div className="compact-input">
                                                <Input label="Pregnancies" name="Pregnancies" type="number" value={inputData.Pregnancies} onChange={handleChange} required placeholder="#" />
                                            </div>
                                            <div className="compact-input">
                                                <Input label="Glucose" name="Glucose" type="number" value={inputData.Glucose} onChange={handleChange} required placeholder="mg/dL" />
                                            </div>

                                            {/* Row 2 */}
                                            <div className="compact-input">
                                                <Input label="Blood Pressure" name="BloodPressure" type="number" value={inputData.BloodPressure} onChange={handleChange} required placeholder="mm Hg" />
                                            </div>
                                            <div className="compact-input">
                                                <Input label="Skin Thickness" name="SkinThickness" type="number" value={inputData.SkinThickness} onChange={handleChange} required placeholder="mm" />
                                            </div>

                                            {/* Row 3 */}
                                            <div className="compact-input">
                                                <Input label="Insulin" name="Insulin" type="number" value={inputData.Insulin} onChange={handleChange} required placeholder="mu U/ml" />
                                            </div>
                                            <div className="compact-input">
                                                <Input label="BMI" name="BMI" type="number" step="0.1" value={inputData.BMI} onChange={handleChange} required placeholder="Value" />
                                            </div>

                                            {/* Row 4 */}
                                            <div className="compact-input">
                                                <Input label="Pedigree Func" name="DiabetesPedigreeFunction" type="number" step="0.01" value={inputData.DiabetesPedigreeFunction} onChange={handleChange} required placeholder="Value" />
                                            </div>
                                            <div className="compact-input">
                                                <Input label="Age" name="Age" type="number" value={inputData.Age} onChange={handleChange} required placeholder="Years" />
                                            </div>
                                        </form>
                                    </div>
                                    <div className="form-actions-fixed">
                                        <Button type="submit" variant="primary" size="lg" isLoading={isLoading} className="w-full glow-button compact-btn">
                                            Assess Risk
                                        </Button>
                                    </div>
                                </Card>
                            ) : (
                                <Card className="result-card fade-in-up" ref={resultRef as any}>
                                    <div className="result-icon pulse-animation">{result.prediction === 0 ? '✓' : '⚠'}</div>
                                    <div className="result-header">Assessment Complete</div>
                                    <div className="result-value" style={{ color: result.prediction === 0 ? 'var(--success)' : 'var(--warning)' }}>
                                        {result.prediction === 0 ? 'Lower Risk' : 'High Risk'}
                                    </div>
                                    <div className="result-probability">
                                        Risk Probability: <span className="highlight">{(result.probability * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="result-guidance compact-guidance">
                                        {result.prediction === 0 ? (
                                            <p>Results suggest a lower risk for diabetes based on the provided metrics.</p>
                                        ) : (
                                            <p>Results indicate a higher risk. Please consult with a healthcare professional.</p>
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
                                <h3><FontAwesomeIcon icon={faInfoCircle} /> About Diabetes</h3>
                                <p>Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. For women, gestational diabetes can also occur during pregnancy.</p>
                            </div>
                            <div className="info-card">
                                <h3><FontAwesomeIcon icon={faUserMd} /> Risk Factors</h3>
                                <ul>
                                    <li><strong>Pregnancy History:</strong> Having gestational diabetes before.</li>
                                    <li><strong>Weight:</strong> Being overweight increases risk.</li>
                                    <li><strong>Family History:</strong> Close relatives with diabetes.</li>
                                    <li><strong>Age:</strong> Risk increases with age.</li>
                                </ul>
                            </div>
                            <div className="info-card">
                                <h3><FontAwesomeIcon icon={faInfoCircle} /> Prevention</h3>
                                <p>Healthy eating, regular physical activity, and maintaining a healthy weight can help prevent type 2 diabetes and manage gestational diabetes risks.</p>
                            </div>
                        </div>

                        <div className="disclaimer-box animate-on-scroll delay-3">
                            <strong>Medical Disclaimer:</strong> This tool is for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
                        </div>
                    </div>
                </div>
            </div>
            <style>{`
                .page-container { display: flex; flex-direction: column; align-items: center; width: 100%; overflow-x: hidden; background: var(--background-dark); }
                .hero-wrapper { height: calc(100vh - 80px); max-height: calc(100vh - 80px); width: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; position: relative; padding: 2rem var(--spacing-lg) 0; overflow: hidden; }
                .hero-split-container { display: flex; flex-direction: column; align-items: center; gap: 1rem; max-width: 1200px; width: 100%; margin: 0 auto; height: 100%; justify-content: center; }
                
                 /* ULTRA COMPACT INPUT OVERRIDES */
                .compact-input input, .compact-input select {
                    padding: 0.3rem 0.5rem !important; /* Tiny padding */
                    font-size: 0.85rem !important;      /* Tiny font */
                    height: 36px !important;            /* Fixed tiny height */
                    line-height: normal;
                    width: 100% !important;
                    background: rgba(255, 255, 255, 0.05) !important; /* Increase visibility */
                    border: 1px solid rgba(255, 255, 255, 0.15) !important;
                    color: var(--text-main) !important;
                    border-radius: 6px !important;
                }
                .compact-input input:focus, .compact-input select:focus {
                     border-color: var(--primary) !important;
                     box-shadow: 0 0 0 2px rgba(236, 72, 153, 0.2) !important;
                     background: rgba(255, 255, 255, 0.1) !important;
                }
                 .compact-input label {
                    font-size: 0.75rem !important;      /* Tiny label */
                    margin-bottom: 0.25rem !important;
                    color: var(--text-secondary) !important;
                    display: block !important;
                    font-weight: 500 !important;
                }
                .compact-input > div { margin-bottom: 0 !important; width: 100% !important; }
                .compact-input { display: flex; flex-direction: column; width: 100%; }

                .card-header-area { padding: 0.75rem 1.25rem 0.5rem; }
                .scrollable-form-box { padding: 0.75rem 1.25rem; }
                .form-actions-fixed { padding: 0.75rem 1.25rem; }
                
                @media (min-width: 1024px) {
                    .hero-split-container { 
                        flex-direction: row; 
                        align-items: flex-start; /* Top align */
                        padding-top: 4rem; 
                        justify-content: space-between; 
                        gap: 2rem; 
                    }
                    .hero-visuals { flex: 1; max-width: 45%; }
                    .hero-form { flex: 1; max-width: 800px; display: flex; justify-content: center; align-items: center; } /* Wider */
                }

                .pulse-circle-container { width: 100px; height: 100px; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: center; position: relative; }

                .visual-content {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    width: 100%;
                }
                .pulse-circle { position: absolute; width: 100%; height: 100%; background: radial-gradient(circle, rgba(236, 72, 153, 0.4) 0%, rgba(0,0,0,0) 70%); border-radius: 50%; animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite; }
                .main-icon { font-size: 3.5rem; color: #ec4899; filter: drop-shadow(0 0 15px rgba(236, 72, 153, 0.6)); animation: float-icon 3s ease-in-out infinite; }
                .page-title { font-size: 3rem; line-height: 1.1; margin-bottom: 0.5rem; font-weight: 700; text-align: center; }
                .text-gradient { background: linear-gradient(135deg, #ec4899 0%, #a855f7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
                .page-subtitle { font-size: 1.1rem; color: var(--text-secondary); margin-bottom: 1.5rem; text-align: center; max-width: 400px; }
                .stats-row { display: flex; gap: 2rem; margin-top: 1rem; }
                .stat-item { display: flex; align-items: center; gap: 0.5rem; color: var(--text-muted); font-size: 0.85rem; }
                .stat-icon { color: var(--primary); }

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
                .card-header-area { padding: 1rem 1.5rem 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05); }
                .form-header { margin: 0; font-size: 1.1rem; color: var(--text-main); }
                .scrollable-form-box { padding: 1.25rem; overflow-y: auto; flex: 1; scrollbar-width: thin; scrollbar-color: var(--primary) rgba(255,255,255,0.1); }
                .scrollable-form-box::-webkit-scrollbar {width: 6px; }
                .scrollable-form-box::-webkit-scrollbar-track {background: rgba(255,255,255,0.05); }
                .scrollable-form-box::-webkit-scrollbar-thumb {background-color: var(--primary); border-radius: 10px; }
                
                .form-actions-fixed { padding: 1rem 1.25rem; background: transparent; /* border-top: 1px solid rgba(255,255,255,0.05); */ }
 
                .compact-grid-form { display: grid; grid-template-columns: 1fr; gap: 0.75rem; width: 100%; }
                .compact-grid-form.four-col { grid-template-columns: repeat(4, 1fr); }
                @media (max-width: 1200px) { .compact-grid-form.four-col { grid-template-columns: repeat(3, 1fr); } }
                @media (max-width: 992px) { .compact-grid-form.four-col { grid-template-columns: repeat(2, 1fr); } }
                @media (max-width: 768px) { .compact-grid-form.four-col { grid-template-columns: 1fr; } }

                @keyframes pulse-ring { 0% { transform: scale(0.5); opacity: 0; } 50% { opacity: 0.5; } 100% { transform: scale(1.5); opacity: 0; } }
                @keyframes float-icon { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
                .fade-in-right { animation: fadeInRight 0.8s ease-out; }
                .fade-in-left { animation: fadeInLeft 0.8s ease-out; }
                .fade-in-up { animation: fadeInUp 0.8s ease-out; }
                @keyframes fadeInRight { from { opacity: 0; transform: translateX(-30px); } to { opacity: 1; transform: translateX(0); } }
                @keyframes fadeInLeft { from { opacity: 0; transform: translateX(30px); } to { opacity: 1; transform: translateX(0); } }
                @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
                
                /* Educational Content Styles */
                .info-panel-wrapper { padding: 4rem var(--spacing-lg); width: 100%; display: flex; flex-direction: column; align-items: center; background: rgba(15, 20, 25, 0.5); }
                .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; width: 100%; margin-bottom: 3rem; }
                .info-card { background: rgba(30, 41, 59, 0.5); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); transition: transform 0.3s ease; }
                .info-card:hover { transform: translateY(-5px); background: rgba(30, 41, 59, 0.7); border-color: var(--primary-light); }
                .info-card h3 { color: var(--primary); font-size: 1.2rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
                .info-card p, .info-card li { color: var(--text-secondary); font-size: 0.95rem; line-height: 1.6; }
                .info-card ul { padding-left: 1.25rem; margin: 0; }
                .info-card li { margin-bottom: 0.5rem; }

                /* Result Card Specific Styles */
                .result-card { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 2rem; min-height: 400px; background: rgba(30, 41, 59, 0.85) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; }
                .result-header { font-size: 1.2rem; margin-bottom: 0.5rem; color: var(--text-light); }
                .result-value { font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem; }
                .result-probability { font-size: 1.1rem; color: var(--text-main); margin-bottom: 1.5rem; }
                .result-icon { font-size: 3rem; margin-bottom: 1rem; color: var(--primary); }
                .compact-guidance p { margin-bottom: 0; color: var(--text-light); max-width: 400px; }
            `}</style>
        </MainLayout>
    );
};

export default WomanDiabetes;
