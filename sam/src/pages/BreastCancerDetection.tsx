import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faRibbon, faMicroscope, faDna, faChartPie, faChartBar, faInfoCircle, faUserMd } from '@fortawesome/free-solid-svg-icons';
import MainLayout from '../components/layout/MainLayout';
import Card from '../components/common/Card';
import Input from '../components/common/Input';
import Button from '../components/common/Button';
import Toast from '../components/common/Toast';
import { API_BASE_URL } from '../utils/api';
import '../styles/global.css';

interface PredictionResult {
    prediction: number;
    probability_breast_cancer: number;
    radar_chart?: string;
    bar_chart?: string;
}

const BreastCancerDetection: React.FC = () => {
    // Initial state with all fields
    const fields = [
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
        'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
        'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error',
        'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
        'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
    ];

    const initialData = fields.reduce((acc, field) => ({ ...acc, [field]: '' }), {});

    // State
    const [inputData, setInputData] = useState<{ [key: string]: string }>(initialData);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
    const [activeTab, setActiveTab] = useState<'radar' | 'bar'>('radar');

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

    const descriptions: { [key: string]: string } = {
        mean_radius: "Mean distance from center to points on the perimeter",
        mean_texture: "Standard deviation of gray-scale values",
        mean_perimeter: "Mean size of the core tumor",
        mean_area: "Mean area of the core tumor",
        mean_smoothness: "Mean values of local variation in radius lengths",
    };

    const getTooltip = (field: string) => {
        if (descriptions[field]) return descriptions[field];
        if (field.includes('error')) return `Standard error of ${field.replace('_error', '').replace(/_/g, ' ')}`;
        if (field.includes('worst')) return `Worst (largest) value for ${field.replace('worst_', '').replace(/_/g, ' ')}`;
        return `Measurement of ${field.replace('mean_', '').replace(/_/g, ' ')}`;
    };

    const formatLabel = (str: string) => {
        return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setInputData({ ...inputData, [e.target.name]: e.target.value });
    };

    const handleReset = () => {
        setResult(null);
        setInputData(initialData);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const validateForm = () => {
        for (const field of fields) {
            const val = Number(inputData[field]);
            if (inputData[field] === '' || isNaN(val)) {
                setToast({ message: `Please enter a valid number for ${formatLabel(field)}`, type: 'error' });
                return false;
            }
            if (val < 0) {
                setToast({ message: `${formatLabel(field)} cannot be negative`, type: 'error' });
                return false;
            }
        }
        return true;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!validateForm()) return;
        setIsLoading(true);

        try {
            const response = await axios.post(`${API_BASE_URL}/detect_breast_cancer`, inputData, { withCredentials: true });
            setResult(response.data);
            setToast({ message: "Screening analysis complete.", type: 'success' });
        } catch (error) {
            console.error('Error:', error);
            setToast({ message: "Error during screening.", type: 'error' });
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
                                        <FontAwesomeIcon icon={faRibbon} className="main-icon" />
                                    </div>
                                </div>
                                <h1 className="page-title">Breast <br /><span className="text-gradient">Screening</span></h1>
                                <p className="page-subtitle">AI analysis of cell nuclei characteristics</p>
                                <div className="stats-row">
                                    <div className="stat-item">
                                        <FontAwesomeIcon icon={faMicroscope} className="stat-icon" />
                                        <span>Cellular Analysis</span>
                                    </div>
                                    <div className="stat-item">
                                        <FontAwesomeIcon icon={faDna} className="stat-icon" />
                                        <span>30+ Biomarkers</span>
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
                                            {fields.map((field) => (
                                                <div key={field} className="compact-input">
                                                    <Input
                                                        label={formatLabel(field)}
                                                        name={field}
                                                        type="number"
                                                        step="any"
                                                        value={inputData[field]}
                                                        onChange={handleChange}
                                                        required
                                                        tooltip={getTooltip(field)}
                                                        className="compact-input-field" // Added class for targeting
                                                    />
                                                </div>
                                            ))}
                                            <div className="form-actions-fixed full-width-col" style={{ gridColumn: '1 / -1', padding: '0.75rem', marginTop: '0.5rem' }}>
                                                <Button onClick={handleSubmit} variant="primary" size="lg" isLoading={isLoading} className="w-full glow-button compact-btn" style={{ fontSize: '1rem', padding: '0.6rem' }}>
                                                    Analyze Cell Data
                                                </Button>
                                            </div>
                                        </form>
                                    </div>
                                </Card>
                            ) : (
                                <Card className="result-card fade-in-up" ref={resultRef}>
                                    <div className="result-header-row">
                                        <div className="result-icon pulse-animation">{result.prediction === 0 ? '⚠' : '✓'}</div>
                                        <div>
                                            <div className="result-title">Screening Result</div>
                                            <div className="result-value" style={{ color: result.prediction === 0 ? 'var(--danger)' : 'var(--success)' }}>
                                                {result.prediction === 0 ? 'Malignant (Cancerous)' : 'Benign (Non-Cancerous)'}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Visualizations Tab */}
                                    {(result.radar_chart || result.bar_chart) && (
                                        <div className="visualization-section">
                                            <div className="viz-tabs">
                                                <button
                                                    className={`viz-tab ${activeTab === 'radar' ? 'active' : ''}`}
                                                    onClick={() => setActiveTab('radar')}
                                                >
                                                    <FontAwesomeIcon icon={faChartPie} /> Radar Map
                                                </button>
                                                <button
                                                    className={`viz-tab ${activeTab === 'bar' ? 'active' : ''}`}
                                                    onClick={() => setActiveTab('bar')}
                                                >
                                                    <FontAwesomeIcon icon={faChartBar} /> Feature Bar
                                                </button>
                                            </div>

                                            <div className="viz-content">
                                                {activeTab === 'radar' && result.radar_chart && (
                                                    <img
                                                        src={`data:image/png;base64,${result.radar_chart}`}
                                                        alt="Radar Chart"
                                                        className="viz-image"
                                                    />
                                                )}
                                                {activeTab === 'bar' && result.bar_chart && (
                                                    <img
                                                        src={`data:image/png;base64,${result.bar_chart}`}
                                                        alt="Bar Chart"
                                                        className="viz-image"
                                                    />
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    <div className="result-probability">
                                        Probability of Malignancy: <span className="highlight">{result.probability_breast_cancer}%</span>
                                    </div>

                                    <div className="result-guidance compact-guidance">
                                        {result.prediction === 0 ? (
                                            <p>Results indicate characteristics of malignancy. Immediate consultation with an oncologist is strongly recommended.</p>
                                        ) : (
                                            <p>Results indicate benign characteristics. Maintain regular screening schedule.</p>
                                        )}
                                    </div>
                                    <Button onClick={handleReset} variant="secondary" className="mt-4 w-full">New Screening</Button>
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
                                <h3><FontAwesomeIcon icon={faRibbon} /> About Breast Cancer</h3>
                                <p>Breast cancer occurs when cells in the breast divide and grow without control. Screening and early detection are crucial for effective treatment.</p>
                            </div>
                            <div className="info-card">
                                <h3><FontAwesomeIcon icon={faUserMd} /> Risk Factors</h3>
                                <ul>
                                    <li><strong>Age:</strong> Risk increases as you get older.</li>
                                    <li><strong>Genetics:</strong> Family history or gene mutations (BRCA1/2).</li>
                                    <li><strong>Lifestyle:</strong> Alcohol consumption and obesity.</li>
                                    <li><strong>History:</strong> Personal history of breast cancer.</li>
                                </ul>
                            </div>
                            <div className="info-card">
                                <h3><FontAwesomeIcon icon={faInfoCircle} /> Prevention</h3>
                                <p>Limit alcohol, maintain a healthy weight, be physically active, and follow recommended screening guidelines for your age and risk level.</p>
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
                        align-items: flex-start;
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
                    /* border-top: 1px solid rgba(255,255,255,0.05); */
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

                 /* Visualization tabs */
                 .visualization-section { margin: 1rem 0; width: 100%; background: rgba(0,0,0,0.2); border-radius: 8px; overflow: hidden; }
                 .viz-tabs { display: flex; border-bottom: 1px solid rgba(255,255,255,0.05); }
                 .viz-tab { flex: 1; padding: 0.5rem; background: transparent; border: none; color: var(--text-muted); cursor: pointer; font-size: 0.85rem; display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
                 .viz-tab.active { color: var(--primary); background: rgba(255,255,255,0.05); border-bottom: 2px solid var(--primary); }
                 .viz-content { padding: 1rem; display: flex; justify-content: center; background: white; }
                 .viz-image { max-width: 100%; height: auto; max-height: 250px; }


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
                    color: #ec4899;
                    filter: drop-shadow(0 0 15px rgba(236, 72, 153, 0.6));
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
                    background: linear-gradient(135deg, #ec4899 0%, #a855f7 100%);
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

export default BreastCancerDetection;
