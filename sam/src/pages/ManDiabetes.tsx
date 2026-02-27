import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faDroplet, faUtensils, faWeight, faInfoCircle, faUserMd } from '@fortawesome/free-solid-svg-icons';
import MainLayout from '../components/layout/MainLayout';
import Card from '../components/common/Card';
import FormField from '../components/common/FormField';
import FormSelect from '../components/common/FormSelect';
import Button from '../components/common/Button';
import Toast from '../components/common/Toast';
import { API_BASE_URL } from '../utils/api';
import '../styles/global.css';

interface PredictionResult {
    prediction: number;
    probability: number;
}

const INITIAL_STATE = {
    Age: '', Hypertension: '', Heart_disease: '', Smoking_history: '',
    BMI: '', HbA1c_level: '', Blood_glucose_level: ''
};

const ManDiabetes: React.FC<{ onLogout?: () => void }> = ({ onLogout }) => {

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

    const getTooltip = (field: string) => {
        const tooltips: Record<string, string> = {
            Age: "Patient's chronological age in years.",
            Hypertension: "Whether the patient has a history of hypertension (high blood pressure).",
            Heart_disease: "Whether the patient has a history of cardiovascular disease.",
            Smoking_history: "Patient's history of smoking tobacco.",
            BMI: "Body mass index (weight in kg / height in m²).",
            HbA1c_level: "Hemoglobin A1c level (%), reflecting average blood sugar over the past 2-3 months.",
            Blood_glucose_level: "Current blood glucose level (mg/dL)."
        };
        return tooltips[field] || "";
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setInputData({ ...inputData, [e.target.name]: e.target.value });
    };

    const handleReset = () => {
        setResult(null);
        setInputData(INITIAL_STATE);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const validateForm = () => {
        const errors: string[] = [];
        const d = inputData;
        if (Number(d.Age) <= 0 || Number(d.Age) > 120) errors.push("Age must be between 1 and 120.");
        if (Number(d.BMI) < 0 || Number(d.BMI) > 70) errors.push("BMI must be between 0 and 70.");
        if (errors.length > 0) {
            setToast({ message: errors[0], type: 'error' });
            return false;
        }
        return true;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!validateForm()) return;
        setIsLoading(true);

        const numericData = {
            Gender: 'Male',
            Age: Number(inputData.Age),
            Hypertension: inputData.Hypertension === 'Yes' ? 1 : 0,
            Heart_disease: inputData.Heart_disease === 'Yes' ? 1 : 0,
            Smoking_history: inputData.Smoking_history,
            BMI: Number(inputData.BMI),
            HbA1c_level: Number(inputData.HbA1c_level),
            Blood_glucose_level: Number(inputData.Blood_glucose_level),
        };

        try {
            const response = await axios.post(`${API_BASE_URL}/predict_male`, { data: numericData });
            setResult(response.data);
            setToast({ message: "Risk assessment completed successfully.", type: 'success' });
        } catch (error) {
            console.error('Error:', error);
            setToast({ message: "Unable to complete assessment. Please try again.", type: 'error' });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <MainLayout isAuthenticated={true} onLogout={onLogout}>

            <div className="page-container">
                {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

                <div className="hero-wrapper">
                    <div className="hero-split-container">
                        <div className="hero-visuals fade-in-right">
                            <div className="visual-content">
                                <div className="pulse-circle-container">
                                    <div className="pulse-circle"></div>
                                    <div className="icon-wrapper">
                                        <FontAwesomeIcon icon={faDroplet} className="main-icon" />
                                    </div>
                                </div>
                                <h1 className="page-title">Diabetes <br /><span className="text-gradient">Risk (Men)</span></h1>
                                <p className="page-subtitle">AI-Driven metabolic health assessment</p>
                                <div className="stats-row">
                                    <div className="stat-item">
                                        <FontAwesomeIcon icon={faUtensils} className="stat-icon" />
                                        <span>Diet Analysis</span>
                                    </div>
                                    <div className="stat-item">
                                        <FontAwesomeIcon icon={faWeight} className="stat-icon" />
                                        <span>BMI Impact</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="hero-form fade-in-left">
                            {!result ? (
                                <Card className="fixed-height-card">
                                    <div className="form-header-sticky">
                                        <h3>Health Assessment</h3>
                                        <p>Enter 7 metabolic and history markers</p>
                                    </div>
                                    <form onSubmit={handleSubmit} className="form-layout-container">
                                        <div className="scrollable-form-box">
                                            <div className="compact-grid-form three-col">
                                                <div className="compact-input">
                                                    <FormField label="Age" name="Age" type="number" tooltip={getTooltip('Age')} value={inputData.Age} onChange={handleChange} required placeholder="Years" />
                                                </div>
                                                <div className="compact-input">
                                                    <FormSelect label="Hypertension" name="Hypertension" tooltip={getTooltip('Hypertension')} value={inputData.Hypertension} onChange={handleChange} options={[
                                                        { value: "0", label: "No" },
                                                        { value: "1", label: "Yes" }
                                                    ]} required />
                                                </div>
                                                <div className="compact-input">
                                                    <FormSelect label="Heart Disease" name="Heart_disease" tooltip={getTooltip('Heart_disease')} value={inputData.Heart_disease} onChange={handleChange} options={[
                                                        { value: "0", label: "No" },
                                                        { value: "1", label: "Yes" }
                                                    ]} required />
                                                </div>
                                                <div className="compact-input">
                                                    <FormSelect label="Smoking History" name="Smoking_history" tooltip={getTooltip('Smoking_history')} value={inputData.Smoking_history} onChange={handleChange} options={[
                                                        { value: "never", label: "Never" },
                                                        { value: "current", label: "Current" },
                                                        { value: "former", label: "Former" },
                                                        { value: "ever", label: "Ever" },
                                                        { value: "not current", label: "Not Current" },
                                                        { value: "No Info", label: "No Info" }
                                                    ]} required />
                                                </div>
                                                <div className="compact-input">
                                                    <FormField label="BMI" name="BMI" type="number" step="0.1" tooltip={getTooltip('BMI')} value={inputData.BMI} onChange={handleChange} required placeholder="Value" />
                                                </div>
                                                <div className="compact-input">
                                                    <FormField label="HbA1c Level" name="HbA1c_level" type="number" step="0.1" tooltip={getTooltip('HbA1c_level')} value={inputData.HbA1c_level} onChange={handleChange} required placeholder="%" />
                                                </div>
                                                <div className="compact-input">
                                                    <FormField label="Blood Glucose" name="Blood_glucose_level" type="number" tooltip={getTooltip('Blood_glucose_level')} value={inputData.Blood_glucose_level} onChange={handleChange} required placeholder="mg/dL" />
                                                </div>
                                            </div>
                                        </div>
                                        <div className="form-actions-sticky">
                                            <Button type="submit" variant="primary" size="lg" isLoading={isLoading} className="w-full glow-button compact-btn" style={{ fontSize: '1rem', padding: '0.6rem' }}>
                                                Assess Risk
                                            </Button>
                                        </div>
                                    </form>
                                </Card>
                            ) : (
                                <Card className="result-card fade-in-up" ref={resultRef}>
                                    <div className="result-header-row">
                                        <div className="result-icon pulse-animation">{result.prediction === 1 ? '⚠' : '✓'}</div>
                                        <div>
                                            <div className="result-title">Assessment Result</div>
                                            <div className="result-value" style={{ color: result.prediction === 1 ? 'var(--danger)' : 'var(--success)' }}>
                                                {result.prediction === 1 ? 'High Risk' : 'Low Risk'}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="result-probability">
                                        Risk Probability: <span className="highlight">{(result.probability * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="result-guidance compact-guidance">
                                        {result.prediction === 1 ? (
                                            <p>Results suggest a higher risk of diabetes. Please consult a healthcare provider for further evaluation.</p>
                                        ) : (
                                            <p>Results are within a lower risk range. Maintain a healthy lifestyle and regular check-ups.</p>
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
                                <p>Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream.</p>
                            </div>
                            <div className="info-card">
                                <h3><FontAwesomeIcon icon={faUserMd} /> Risk Factors</h3>
                                <ul>
                                    <li><strong>Weight:</strong> Being overweight increases risk.</li>
                                    <li><strong>Age:</strong> Risk increases as you get older.</li>
                                    <li><strong>Inactivity:</strong> Being active less than 3 times a week.</li>
                                    <li><strong>Genetics:</strong> Family history of diabetes.</li>
                                </ul>
                            </div>
                            <div className="info-card">
                                <h3><FontAwesomeIcon icon={faInfoCircle} /> Prevention</h3>
                                <p>Type 2 diabetes can be prevented or delayed with healthy lifestyle changes. Losing a small amount of weight and getting regular physical activity can prevent or delay type 2 diabetes.</p>
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
                        padding-top: 4rem;
                        justify-content: space-between;
                        justify-content: space-between;
                        gap: 2rem;
                    }

                    .hero-visuals {
                        flex: 1;
                        max-width: 45%;
                        max-width: 45%;
                        display: flex;
                        flex-direction: column;
                        /* justify-content: center; */
                    }

                    .hero-form {
                        flex: 1;
                        max-width: 800px; /* Wider for 4-col */
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
                    height: 600px;
                    max-height: 80vh;
                    width: 100%;
                    overflow: hidden;
                    border-radius: 12px !important;
                }

                .form-header-sticky {
                    padding: 1.25rem 1.25rem 0.5rem;
                    border-bottom: 1px solid rgba(255,255,255,0.05);
                    background: rgb(30, 41, 59);
                    z-index: 10;
                }
                .form-header-sticky h3 { margin: 0; font-size: 1.1rem; color: var(--text-main); }
                .form-header-sticky p { margin: 0.25rem 0 0 0; font-size: 0.8rem; color: var(--text-secondary); }

                .form-layout-container {
                    display: flex;
                    flex-direction: column;
                    flex: 1;
                    overflow: hidden;
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

                .compact-grid-form {
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 1.25rem 0.75rem; 
                    width: 100%;
                }

                .compact-grid-form.three-col { grid-template-columns: repeat(3, 1fr); }

                @media (max-width: 1024px) { .compact-grid-form.three-col { grid-template-columns: repeat(2, 1fr); } }
                @media (max-width: 640px) { .compact-grid-form.three-col { grid-template-columns: 1fr; } }

                .form-actions-sticky {
                    padding: 1rem 1.25rem;
                    background: rgba(30, 41, 59, 1);
                    border-top: 1px solid rgba(255,255,255,0.1);
                    z-index: 10;
                    margin-top: auto;
                }

                .compact-input {
                    display: flex;
                    flex-direction: column;
                    width: 100%;
                    height: 100%;
                }

                .compact-input > div { margin-bottom: 0 !important; width: 100% !important; }

                .compact-input input, .compact-input select {
                    padding: 0.3rem 0.5rem !important;
                    font-size: 0.85rem !important;
                    height: 36px !important;
                    line-height: normal;
                    width: 100% !important;
                    background: rgba(255, 255, 255, 0.05) !important;
                    border: 1px solid rgba(255, 255, 255, 0.15) !important;
                    color: var(--text-main) !important;
                    border-radius: 6px !important;
                }
                
                .compact-input input:focus, .compact-input select:focus {
                     border-color: var(--primary) !important;
                     box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
                     background: rgba(255, 255, 255, 0.1) !important;
                }

                .compact-input label {
                    font-size: 0.75rem !important;
                    margin-bottom: 0.25rem !important;
                    color: var(--text-secondary) !important;
                    display: block !important;
                    font-weight: 500 !important;
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

                .pulse-circle {
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    background: radial-gradient(circle, rgba(99, 102, 241, 0.4) 0%, rgba(0,0,0,0) 70%);
                    border-radius: 50%;
                    animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
                }

                .main-icon {
                    font-size: 3.5rem;
                    color: #6366f1;
                    filter: drop-shadow(0 0 15px rgba(99, 102, 241, 0.6));
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
                    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
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
            `}</style>
        </MainLayout>
    );
};

export default ManDiabetes;
