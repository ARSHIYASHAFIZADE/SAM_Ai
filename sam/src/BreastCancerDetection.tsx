import React, { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './utils/api';
import styles from './BreastCancerDetection.module.css';
import FormGrid from './components/common/FormGrid';
import FormField from './components/common/FormField';
import Input from './components/common/Input';
import SectionTitle from './components/common/SectionTitle';

interface InputData {
    [key: string]: string;
}

interface PredictionResult {
    prediction: number;
    probability_breast_cancer: string;
}

const formatLabel = (key: string) => key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

const BreastCancerDetection: React.FC = () => {
    const [inputData, setInputData] = useState<InputData>({
        mean_radius: '',
        mean_texture: '',
        mean_perimeter: '',
        mean_area: '',
        mean_smoothness: '',
        mean_compactness: '',
        mean_concavity: '',
        mean_concave_points: '',
        mean_symmetry: '',
        mean_fractal_dimension: '',
        radius_error: '',
        texture_error: '',
        perimeter_error: '',
        area_error: '',
        smoothness_error: '',
        compactness_error: '',
        concavity_error: '',
        concave_points_error: '',
        symmetry_error: '',
        fractal_dimension_error: '',
        worst_radius: '',
        worst_texture: '',
        worst_perimeter: '',
        worst_area: '',
        worst_smoothness: '',
        worst_compactness: '',
        worst_concavity: '',
        worst_concave_points: '',
        worst_symmetry: '',
        worst_fractal_dimension: ''
    });

    const [result, setResult] = useState<PredictionResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [blur, setBlur] = useState(false);

    const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setInputData(prevData => ({
            ...prevData,
            [name]: value
        }));
    };

    const isValidNumber = (value: string): boolean => {
        return !isNaN(Number(value)) && Number(value) >= 0;
    };

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const invalidEntries = Object.values(inputData).some(value => !isValidNumber(value));
        if (invalidEntries) {
            setError('Please enter valid numbers for all fields.');
            return;
        }

        try {
            const response = await axios.post(`${API_BASE_URL}/detect_breast_cancer`, inputData, { withCredentials: true });
            setResult(response.data);
            setError(null);
            setBlur(true);
        } catch (error) {
            console.error('Error:', error);
            setError('An error occurred while submitting the form. Please try again later.');
            setResult(null);
            setBlur(false);
        }
    };

    const handleCancel = () => {
        setResult(null);
        setError(null);
        setBlur(false);
    };

    return (
        <div className={styles.bc_wrapper}>
            {/* Light animation effect */}
            <div className={styles.bc_lightEffect}></div>

            {/* Form Container */}
            <div className={`${styles.bc_formContainer} ${blur ? styles.blur : ''}`}>
                <h1 className={styles.bc_h1} style={{ marginBottom: '2rem' }}>Breast Cancer Detection</h1>
                <form onSubmit={handleSubmit}>
                    <FormGrid>
                        <SectionTitle title="Mean Characteristics" />
                        {Object.keys(inputData).filter(k => k.startsWith('mean_')).map((key) => (
                            <FormField key={key}>
                                <Input
                                    label={formatLabel(key)}
                                    type="number"
                                    id={key}
                                    name={key}
                                    placeholder={`Enter ${key.replace(/_/g, ' ')}`}
                                    value={inputData[key]}
                                    onChange={handleChange}
                                    required
                                    step="any"
                                    tooltip={`Mean measurement for ${formatLabel(key.replace('mean_', '')).toLowerCase()}.`}
                                />
                            </FormField>
                        ))}
                        
                        <SectionTitle title="Error Characteristics" />
                        {Object.keys(inputData).filter(k => k.includes('_error')).map((key) => (
                            <FormField key={key}>
                                <Input
                                    label={formatLabel(key)}
                                    type="number"
                                    id={key}
                                    name={key}
                                    placeholder={`Enter ${key.replace(/_/g, ' ')}`}
                                    value={inputData[key]}
                                    onChange={handleChange}
                                    required
                                    step="any"
                                    tooltip={`Standard error for ${formatLabel(key.replace('_error', '')).toLowerCase()}.`}
                                />
                            </FormField>
                        ))}

                        <SectionTitle title="Worst Characteristics" />
                        {Object.keys(inputData).filter(k => k.startsWith('worst_')).map((key) => (
                            <FormField key={key}>
                                <Input
                                    label={formatLabel(key)}
                                    type="number"
                                    id={key}
                                    name={key}
                                    placeholder={`Enter ${key.replace(/_/g, ' ')}`}
                                    value={inputData[key]}
                                    onChange={handleChange}
                                    required
                                    step="any"
                                    tooltip={`Worst/largest measurement for ${formatLabel(key.replace('worst_', '')).toLowerCase()}.`}
                                />
                            </FormField>
                        ))}

                        <FormField fullWidth>
                            <button type="submit" className="medical-submit-btn" style={{ marginBottom: '2rem' }}>Submit</button>
                        </FormField>
                    </FormGrid>

                </form>
            </div>

            {/* Result Card */}
            {result && (
                <div className={`${styles.bc_resultCard} ${result ? styles.show : ''}`}>
                    <h2>Result</h2>
                    <p>{result.prediction === 0 ? 'Breast cancer is Malignant' : 'Breast cancer is Benign'}</p>
                    {result.probability_breast_cancer && (
                        <p>Probability of having Malignant breast cancer: {result.probability_breast_cancer}%</p>
                    )}
                    <button onClick={handleCancel} className={styles.bc_cancelBtn}>Cancel</button>
                </div>
            )}

            {/* Error Card */}
            {error && (
                <div className={`${styles.bc_errorCard} ${error ? styles.show : ''}`}>
                    <p>{error}</p>
                </div>
            )}
        </div>
    );
};

export default BreastCancerDetection;
