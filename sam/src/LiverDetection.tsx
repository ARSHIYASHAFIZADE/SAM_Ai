import React, { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './utils/api';
import styles from './LiverDetection.module.css';

interface PredictionResult {
    prediction: number; 
    probability_healthy_liver: string; 
}

const LiverDetection: React.FC = () => {
    const [inputData, setInputData] = useState({
        Age: '',
        Gender: '',
        BMI: '',
        AlcoholConsumption: '',
        Smoking: '',
        GeneticRisk: '',
        PhysicalActivity: '',
        Diabetes: '',
        Cholesterol: '',
        LiverFunctionTest: ''
    });

    const [result, setResult] = useState<PredictionResult | null>(null);
    const [blur, setBlur] = useState(false);

    const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setInputData({
            ...inputData,
            [name]: value
        });
    };

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const numericData = {
            Age: Number(inputData.Age),
            Gender: Number(inputData.Gender),
            BMI: Number(inputData.BMI),
            AlcoholConsumption: Number(inputData.AlcoholConsumption),
            Smoking: Number(inputData.Smoking),
            GeneticRisk: Number(inputData.GeneticRisk),
            PhysicalActivity: Number(inputData.PhysicalActivity),
            Diabetes: Number(inputData.Diabetes),
            Cholesterol: Number(inputData.Cholesterol),
            LiverFunctionTest: Number(inputData.LiverFunctionTest)
        };

        const formattedData = [
            numericData.Age,
            numericData.Gender,
            numericData.BMI,
            numericData.AlcoholConsumption,
            numericData.Smoking,
            numericData.GeneticRisk,
            numericData.PhysicalActivity,
            numericData.Diabetes,
            numericData.Cholesterol,
            numericData.LiverFunctionTest
        ];

        try {
            const response = await axios.post(`${API_BASE_URL}/detect_liver`, {
                input_data: formattedData
            });
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
                <h1>Liver Health Detection</h1>
                <form id="liverForm" onSubmit={handleSubmit}>
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
                        <label htmlFor="Gender">Gender:</label>
                        <select
                            id="Gender"
                            name="Gender"
                            value={inputData.Gender}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Gender</option>
                            <option value="0">Male</option>
                            <option value="1">Female</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="BMI">BMI:</label>
                        <input
                            type="number"
                            id="BMI"
                            name="BMI"
                            step="0.01"
                            placeholder="BMI (e.g., 22.5)"
                            value={inputData.BMI}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="AlcoholConsumption">Alcohol Consumption:</label>
                        <input
                            type="number"
                            id="AlcoholConsumption"
                            name="AlcoholConsumption"
                            step="0.01"
                            placeholder="Alcohol Consumption (Units/Week)"
                            value={inputData.AlcoholConsumption}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Smoking">Smoking Status:</label>
                        <select
                            id="Smoking"
                            name="Smoking"
                            value={inputData.Smoking}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Status</option>
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="GeneticRisk">Genetic Risk:</label>
                        <select
                            id="GeneticRisk"
                            name="GeneticRisk"
                            value={inputData.GeneticRisk}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Risk Level</option>
                            <option value="0">Low</option>
                            <option value="1">Medium</option>
                            <option value="2">High</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="PhysicalActivity">Physical Activity (Hrs/Week):</label>
                        <input
                            type="number"
                            id="PhysicalActivity"
                            name="PhysicalActivity"
                            step="0.01"
                            placeholder="Physical Activity"
                            value={inputData.PhysicalActivity}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Diabetes">Diabetes Status:</label>
                        <select
                            id="Diabetes"
                            name="Diabetes"
                            value={inputData.Diabetes}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Select Status</option>
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Cholesterol">Cholesterol Level:</label>
                        <input
                            type="number"
                            id="Cholesterol"
                            name="Cholesterol"
                            step="0.01"
                            placeholder="Cholesterol"
                            value={inputData.Cholesterol}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="LiverFunctionTest">Liver Function Test Score:</label>
                        <input
                            type="number"
                            id="LiverFunctionTest"
                            name="LiverFunctionTest"
                            step="0.01"
                            placeholder="Liver Function Test"
                            value={inputData.LiverFunctionTest}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <button type="submit" className={styles.submitBtn}>Submit</button>
                </form>
            </div>
            {result && (
                <div className={styles.resultCard}>
                    <h2 className={styles.r}>Result</h2>
                    <p className={styles.r}>Prediction: {result.prediction === 1 ? 'Liver Problem Detected' : 'No Liver Problems'}</p>
                    <p className={styles.r}>Confidence: {result.probability_healthy_liver}%</p>
                    <button onClick={handleCancel} className={styles.cancelBtn}>Cancel</button>
                </div>
            )}
        </div>
    );
};

export default LiverDetection;
