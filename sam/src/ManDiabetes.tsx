import React, { useState } from 'react';
import axios from 'axios';
import styles from './ManDiabetes.module.css';

interface PredictionResult {
    prediction: number;
    probability: number;
}

const ManDiabetes = () => {
    const [inputData, setInputData] = useState({
        Age: '',
        Hypertension: '',
        Heart_disease: '',
        Smoking_history: '',
        BMI: '',
        HbA1c_level: '',
        Blood_glucose_level: '',
    });

    const [result, setResult] = useState<PredictionResult | null>(null);
    const [blur, setBlur] = useState(false);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement> | React.ChangeEvent<HTMLSelectElement>) => {
        setInputData({
            ...inputData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const numericData = {
            Gender: 'Male', // Hardcoded as Male
            Age: Number(inputData.Age),
            Hypertension: inputData.Hypertension === 'Yes' ? 1 : 0, // Maps Yes to 1 and No to 0
            Heart_disease: inputData.Heart_disease === 'Yes' ? 1 : 0, // Maps Yes to 1 and No to 0
            Smoking_history: inputData.Smoking_history,
            BMI: Number(inputData.BMI),
            HbA1c_level: Number(inputData.HbA1c_level),
            Blood_glucose_level: Number(inputData.Blood_glucose_level),
        };

        try {
            const response = await axios.post('https://sam-ai-mu6e.onrender.com/predict_male', { data: numericData });
            setResult(response.data);
            setBlur(true);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while submitting the form. Please try again later.');
        }
    };

    const handleCancel = () => {
        setResult(null);
        setBlur(false);
    };

    return (
        <div className={styles.formWrapper}>
            <div className={`${styles.container} ${blur ? styles.blur : ''}`}>
                <h1>Diabetes Detection for Men</h1>
                <form id="diabetesForm" onSubmit={handleSubmit}>
                    <div className={styles.formGroup}>
                        <label htmlFor="Age">Age:</label>
                        <input
                            type="number"
                            id="Age"
                            name="Age"
                            placeholder="Age"
                            value={inputData.Age}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Hypertension">Hypertension:</label>
                        <select
                            id="Hypertension"
                            name="Hypertension"
                            value={inputData.Hypertension}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Hypertension Status</option>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Heart_disease">Heart Disease:</label>
                        <select
                            id="Heart_disease"
                            name="Heart_disease"
                            value={inputData.Heart_disease}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Heart Disease Status</option>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Smoking_history">Smoking History:</label>
                        <select
                            id="Smoking_history"
                            name="Smoking_history"
                            value={inputData.Smoking_history}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Smoking History</option>
                            <option value="never">Never</option>
                            <option value="current">Current</option>
                            <option value="former">Former</option>
                            <option value="ever">Ever</option>
                            <option value="No Info">No Info</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="BMI">BMI:</label>
                        <input
                            type="number"
                            id="BMI"
                            name="BMI"
                            placeholder="BMI"
                            value={inputData.BMI}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="HbA1c_level">HbA1c Level:</label>
                        <input
                            type="number"
                            id="HbA1c_level"
                            name="HbA1c_level"
                            placeholder="HbA1c Level"
                            value={inputData.HbA1c_level}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Blood_glucose_level">Blood Glucose Level:</label>
                        <input
                            type="number"
                            id="Blood_glucose_level"
                            name="Blood_glucose_level"
                            placeholder="Blood Glucose Level"
                            value={inputData.Blood_glucose_level}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <button type="submit" className={styles.submitBtn}>Submit</button>
                </form>
            </div>
            {result && (
                <div className={styles.resultCard}>
                    <h2 className={styles.pr}>Result</h2>
                    <p className={styles.pr}>Prediction: {result.prediction === 0 ? 'No Diabetes' : 'Diabetes'}</p>
                    <p className={styles.pr}>Probability: {result.probability}%</p>
                    <button onClick={handleCancel} className={styles.cancelBtn}>Cancel</button>
                </div>
            )}
        </div>
    );
};

export default ManDiabetes;
