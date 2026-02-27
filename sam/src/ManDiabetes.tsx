import React, { useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './utils/api';
import styles from './ManDiabetes.module.css';
import FormGrid from './components/common/FormGrid';
import FormField from './components/common/FormField';
import Input from './components/common/Input';
import Select from './components/common/Select';
import SectionTitle from './components/common/SectionTitle';

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

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setInputData({
            ...inputData,
            [e.target.name as keyof typeof inputData]: e.target.value
        });
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const numericData = {
            Gender: 'Male', // Hardcoded as Male
            Age: Number(inputData.Age),
            Hypertension: inputData.Hypertension === 'Yes' ? 1 : 0,
            Heart_disease: inputData.Heart_disease === 'Yes' ? 1 : 0,
            Smoking_history: inputData.Smoking_history,
            bmi: Number(inputData.BMI),
            HbA1c_level: Number(inputData.HbA1c_level),
            blood_glucose_level: Number(inputData.Blood_glucose_level),
        };

        try {
            const response = await axios.post(`${API_BASE_URL}/predict_male`, { data: numericData });
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
                <h1 style={{ marginBottom: '2rem' }}>Diabetes Detection for Men</h1>
                <form id="diabetesForm" onSubmit={handleSubmit}>
                    <FormGrid>
                        <SectionTitle title="Demographics" />
                        <FormField fullWidth>
                            <Input
                                label="Age"
                                type="number"
                                id="Age"
                                name="Age"
                                placeholder="Years"
                                value={inputData.Age}
                                onChange={handleChange}
                                required
                                tooltip="Patient's age in years."
                            />
                        </FormField>

                        <SectionTitle title="Medical History & Lifestyle" />
                        <FormField>
                            <Select
                                label="Hypertension"
                                id="Hypertension"
                                name="Hypertension"
                                value={inputData.Hypertension}
                                onChange={handleChange}
                                required
                                options={[
                                    { value: 'Yes', label: 'Yes' },
                                    { value: 'No', label: 'No' },
                                ]}
                                tooltip="Does the patient have hypertension?"
                            />
                        </FormField>
                        <FormField>
                            <Select
                                label="Heart Disease"
                                id="Heart_disease"
                                name="Heart_disease"
                                value={inputData.Heart_disease}
                                onChange={handleChange}
                                required
                                options={[
                                    { value: 'Yes', label: 'Yes' },
                                    { value: 'No', label: 'No' },
                                ]}
                                tooltip="Does the patient have a history of heart disease?"
                            />
                        </FormField>
                        <FormField fullWidth>
                            <Select
                                label="Smoking History"
                                id="Smoking_history"
                                name="Smoking_history"
                                value={inputData.Smoking_history}
                                onChange={handleChange}
                                required
                                options={[
                                    { value: 'never', label: 'Never' },
                                    { value: 'current', label: 'Current' },
                                    { value: 'former', label: 'Former' },
                                    { value: 'ever', label: 'Ever' },
                                    { value: 'No Info', label: 'No Info' },
                                ]}
                                tooltip="Patient's smoking history."
                            />
                        </FormField>

                        <SectionTitle title="Measurements & Lab Values" />
                        <FormField>
                            <Input
                                label="BMI (Body Mass Index)"
                                type="number"
                                id="BMI"
                                name="BMI"
                                step="0.01"
                                placeholder="e.g., 22.5"
                                value={inputData.BMI}
                                onChange={handleChange}
                                required
                                tooltip="Body Mass Index."
                            />
                        </FormField>
                        <FormField>
                            <Input
                                label="HbA1c Level"
                                type="number"
                                id="HbA1c_level"
                                name="HbA1c_level"
                                placeholder="%"
                                step="0.1"
                                value={inputData.HbA1c_level}
                                onChange={handleChange}
                                required
                                tooltip="Hemoglobin A1c level in percentage."
                            />
                        </FormField>
                        <FormField fullWidth>
                            <Input
                                label="Blood Glucose Level"
                                type="number"
                                id="Blood_glucose_level"
                                name="Blood_glucose_level"
                                placeholder="mg/dl"
                                value={inputData.Blood_glucose_level}
                                onChange={handleChange}
                                required
                                tooltip="Blood glucose level in mg/dl."
                            />
                        </FormField>

                        <FormField fullWidth>
                            <button type="submit" className="medical-submit-btn">Submit</button>
                        </FormField>
                    </FormGrid>

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
