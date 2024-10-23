import React, { useState } from 'react';
import axios from 'axios';
import styles from './HeartDisease.module.css';

interface PredictionResult {
    prediction: number;
    probability: string; 
}


const HeartDisease: React.FC = () => {
    const [inputData, setInputData] = useState({
        age: '',
        sex: '',
        cp: '',
        trestbps: '',
        chol: '',
        fbs: '',
        restecg: '',
        thalach: '',
        exang: '',
        oldpeak: '',
        slope: '',
        ca: '',
        thal: ''
    });

    const [result, setResult] = useState<PredictionResult | null>(null);
    const [blur, setBlur] = useState(false);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setInputData({
            ...inputData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const numericData = {
            age: Number(inputData.age),
            sex: Number(inputData.sex),
            cp: Number(inputData.cp),
            trestbps: Number(inputData.trestbps),
            chol: Number(inputData.chol),
            fbs: Number(inputData.fbs),
            restecg: Number(inputData.restecg),
            thalach: Number(inputData.thalach),
            exang: Number(inputData.exang),
            oldpeak: Number(inputData.oldpeak),
            slope: Number(inputData.slope),
            ca: Number(inputData.ca),
            thal: Number(inputData.thal),
        };
    
        try {
            const response = await axios.post('https://sam-ai-mu6e.onrender.com/detect_heart', { data: numericData });
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
                <h1>Heart Disease Detection</h1>
                <form id="heartDiseaseForm" onSubmit={handleSubmit}>
                    {/* Form fields */}
                    <div className={styles.formGroup}>
                        <label htmlFor="Age">Age:</label>
                        <input
                            type="number"
                            id="Age"
                            name="age"
                            placeholder="Age"
                            value={inputData.age}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Sex">Sex:</label>
                        <select
                            id="Sex"
                            name="sex"
                            value={inputData.sex}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Sex</option>
                            <option value="0">Female</option>
                            <option value="1">Male</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="ChestPain">Chest Pain Type:</label>
                        <select
                            id="ChestPain"
                            name="cp"
                            value={inputData.cp}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Chest Pain Type</option>
                            <option value="1">Typical Angina</option>
                            <option value="2">Atypical Angina</option>
                            <option value="3">Non-Anginal Pain</option>
                            <option value="4">Asymptomatic</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="RestingBP">Resting Blood Pressure:</label>
                        <input
                            type="number"
                            id="RestingBP"
                            name="trestbps"
                            placeholder="Resting Blood Pressure"
                            value={inputData.trestbps}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Cholesterol">Serum Cholesterol:</label>
                        <input
                            type="number"
                            id="Cholesterol"
                            name="chol"
                            placeholder="Serum Cholesterol"
                            value={inputData.chol}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="FastingBS">Fasting Blood Sugar:</label>
                        <select
                            id="FastingBS"
                            name="fbs"
                            value={inputData.fbs}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Fasting Blood Sugar</option>
                            <option value="0">Less than 120 mg/dl</option>
                            <option value="1">Greater than 120 mg/dl</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="RestingECG">Resting ECG:</label>
                        <select
                            id="RestingECG"
                            name="restecg"
                            value={inputData.restecg}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Resting ECG Result</option>
                            <option value="0">Normal</option>
                            <option value="1">Having ST-T Wave Abnormality</option>
                            <option value="2">Showing Probable or Definite Left Ventricular Hypertrophy</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="MaxHR">Maximum Heart Rate:</label>
                        <input
                            type="number"
                            id="MaxHR"
                            name="thalach"
                            placeholder="Maximum Heart Rate"
                            value={inputData.thalach}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="ExerciseAngina">Exercise Induced Angina:</label>
                        <select
                            id="ExerciseAngina"
                            name="exang"
                            value={inputData.exang}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Exercise Induced Angina</option>
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Oldpeak">Oldpeak:</label>
                        <input
                            type="number"
                            id="Oldpeak"
                            name="oldpeak"
                            placeholder="Oldpeak"
                            value={inputData.oldpeak}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Slope">Slope of Peak Exercise ST Segment:</label>
                        <select
                            id="Slope"
                            name="slope"
                            value={inputData.slope}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Slope</option>
                            <option value="1">Upsloping</option>
                            <option value="2">Flat</option>
                            <option value="3">Downsloping</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="CA">Number of Major Vessels:</label>
                        <input
                            type="number"
                            id="CA"
                            name="ca"
                            placeholder="Number of Major Vessels"
                            value={inputData.ca}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Thal">Thalassemia:</label>
                        <select
                            id="Thal"
                            name="thal"
                            value={inputData.thal}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Thalassemia</option>
                            <option value="3">Normal</option>
                            <option value="6">Fixed Defect</option>
                            <option value="7">Reversible Defect</option>
                        </select>
                    </div>
                    <button type="submit" className={styles.submitBtn}>Submit</button>
                </form>
            </div>
            {result && (
                <div className={styles.resultCard}>
                    <h2 className={styles.r}>Result</h2>
                    <p className={styles.r}>Prediction: {result.prediction === 0 ? 'No Heart Disease' : 'Heart Disease'}</p>
                    <p className={styles.r}>Probability: {result.probability}%</p>
                    <button onClick={handleCancel} className={styles.cancelBtn}>Cancel</button>
                </div>
            )}

        </div>
    );
};

export default HeartDisease;
