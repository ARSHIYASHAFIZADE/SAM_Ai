import React, { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';
import styles from './BreastCancerDetection.module.css';

interface InputData {
    [key: string]: string;
}

interface PredictionResult {
    prediction: number;
    probability_breast_cancer: string;
}

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
            const response = await axios.post('https://sam-ai-mu6e.onrender.com/detect_breast_cancer', inputData, { withCredentials: true });
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
                <h1 className={styles.bc_h1}>Breast Cancer Detection</h1>
                <form onSubmit={handleSubmit}>
                    {Object.keys(inputData).map((key) => (
                        <div className={styles.bc_formGroup} key={key}>
                            <label htmlFor={key} className={styles.bc_label}>{key.replace(/_/g, ' ')}:</label>
                            <input
                                type="number"
                                id={key}
                                name={key}
                                placeholder={`Enter ${key.replace(/_/g, ' ')}`}
                                value={inputData[key]}
                                onChange={handleChange}
                                required
                                className={styles.bc_input}
                            />
                        </div>
                    ))}
                    <button type="submit" className={styles.bc_submitBtn}>Submit</button>
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
