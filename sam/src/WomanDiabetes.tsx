import React, { useState } from 'react';
import axios from 'axios';
import styles from './WomanDiabetes.module.css';

interface PredictionResult {
    prediction: number; 
    probability: number; 
}

const WomanDiabetes = () => {
    const [inputData, setInputData] = useState({
        Pregnancies: '',
        Glucose: '',
        BloodPressure: '',
        SkinThickness: '',
        Insulin: '',
        BMI: '',
        DiabetesPedigreeFunction: '',
        Age: ''
    });

    const [result, setResult] = useState<PredictionResult | null>(null);
    const [blur, setBlur] = useState(false);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setInputData({
            ...inputData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
    
        // Add all required fields
        const numericData = {
            Pregnancies: Number(inputData.Pregnancies), 
            Glucose: Number(inputData.Glucose),
            BloodPressure: Number(inputData.BloodPressure),
            SkinThickness: Number(inputData.SkinThickness),
            Insulin: Number(inputData.Insulin),
            BMI: Number(inputData.BMI),
            DiabetesPedigreeFunction: Number(inputData.DiabetesPedigreeFunction),
            Age: Number(inputData.Age),
            // Add other required fields if necessary
        };
    
        try {
            const response = await axios.post('https://sam-ai-mu6e.onrender.com/api/predict', { data: numericData });
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
                <h1>Diabetes Detection for woman</h1>
                <form id="diabetesForm" onSubmit={handleSubmit}>
                    {/* Form fields */}
                    <div className={styles.formGroup}>
                        <label htmlFor="Pregnancies">Pregnancies:</label>
                        <input
                            type="number"
                            id="Pregnancies"
                            name="Pregnancies"
                            placeholder="Pregnancies"
                            value={inputData.Pregnancies}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Glucose">Glucose:</label>
                        <input
                            type="number"
                            id="Glucose"
                            name="Glucose"
                            placeholder="Glucose"
                            value={inputData.Glucose}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="BloodPressure">Blood Pressure:</label>
                        <input
                            type="number"
                            id="BloodPressure"
                            name="BloodPressure"
                            placeholder="Blood Pressure"
                            value={inputData.BloodPressure}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="SkinThickness">Skin Thickness:</label>
                        <input
                            type="number"
                            id="SkinThickness"
                            name="SkinThickness"
                            placeholder="Skin Thickness"
                            value={inputData.SkinThickness}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Insulin">Insulin:</label>
                        <input
                            type="number"
                            id="Insulin"
                            name="Insulin"
                            placeholder="Insulin"
                            value={inputData.Insulin}
                            onChange={handleChange}
                            required
                        />
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
                        <label htmlFor="DiabetesPedigreeFunction">Diabetes Pedigree Function:</label>
                        <input
                            type="number"
                            id="DiabetesPedigreeFunction"
                            name="DiabetesPedigreeFunction"
                            placeholder="Diabetes Pedigree Function"
                            value={inputData.DiabetesPedigreeFunction}
                            onChange={handleChange}
                            required
                        />
                    </div>
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

export default WomanDiabetes;
