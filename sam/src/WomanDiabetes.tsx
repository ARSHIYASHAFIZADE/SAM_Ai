import React, { useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './utils/api';
import styles from './WomanDiabetes.module.css';
import FormGrid from './components/common/FormGrid';
import FormField from './components/common/FormField';
import Input from './components/common/Input';
import SectionTitle from './components/common/SectionTitle';

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
            [e.target.name as keyof typeof inputData]: e.target.value
        });
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
    
        const numericData = {
            Pregnancies: Number(inputData.Pregnancies), 
            Glucose: Number(inputData.Glucose),
            BloodPressure: Number(inputData.BloodPressure),
            SkinThickness: Number(inputData.SkinThickness),
            Insulin: Number(inputData.Insulin),
            BMI: Number(inputData.BMI),
            DiabetesPedigreeFunction: Number(inputData.DiabetesPedigreeFunction),
            Age: Number(inputData.Age),
        };
    
        try {
            const response = await axios.post(`${API_BASE_URL}/predict`, { data: numericData });
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
                <h1 style={{ marginBottom: '2rem' }}>Diabetes Detection for Woman</h1>
                <form id="diabetesForm" onSubmit={handleSubmit}>
                    <FormGrid>
                        <SectionTitle title="Demographics" />
                        <FormField>
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
                        <FormField>
                            <Input
                                label="Pregnancies"
                                type="number"
                                id="Pregnancies"
                                name="Pregnancies"
                                placeholder="Number of Pregnancies"
                                value={inputData.Pregnancies}
                                onChange={handleChange}
                                required
                                tooltip="Number of times pregnant."
                            />
                        </FormField>

                        <SectionTitle title="Measurements & Vital Signs" />
                        <FormField>
                            <Input
                                label="BMI (Body Mass Index)"
                                type="number"
                                id="BMI"
                                name="BMI"
                                placeholder="e.g., 22.5"
                                value={inputData.BMI}
                                onChange={handleChange}
                                required
                                tooltip="Body mass index (weight in kg/(height in m)^2)."
                            />
                        </FormField>
                        <FormField>
                            <Input
                                label="Blood Pressure"
                                type="number"
                                id="BloodPressure"
                                name="BloodPressure"
                                placeholder="mm Hg"
                                value={inputData.BloodPressure}
                                onChange={handleChange}
                                required
                                tooltip="Diastolic blood pressure (mm Hg)."
                            />
                        </FormField>
                        <FormField fullWidth>
                            <Input
                                label="Skin Thickness"
                                type="number"
                                id="SkinThickness"
                                name="SkinThickness"
                                placeholder="mm"
                                value={inputData.SkinThickness}
                                onChange={handleChange}
                                required
                                tooltip="Triceps skin fold thickness (mm)."
                            />
                        </FormField>

                        <SectionTitle title="Lab Values & Risk Factors" />
                        <FormField>
                            <Input
                                label="Glucose"
                                type="number"
                                id="Glucose"
                                name="Glucose"
                                placeholder="mg/dl"
                                value={inputData.Glucose}
                                onChange={handleChange}
                                required
                                tooltip="Plasma glucose concentration a 2 hours in an oral glucose tolerance test."
                            />
                        </FormField>
                        <FormField>
                            <Input
                                label="Insulin"
                                type="number"
                                id="Insulin"
                                name="Insulin"
                                placeholder="IU/ml"
                                value={inputData.Insulin}
                                onChange={handleChange}
                                required
                                tooltip="2-Hour serum insulin (mu U/ml)."
                            />
                        </FormField>
                        <FormField fullWidth>
                            <Input
                                label="Diabetes Pedigree Function"
                                type="number"
                                id="DiabetesPedigreeFunction"
                                name="DiabetesPedigreeFunction"
                                placeholder="e.g., 0.627"
                                value={inputData.DiabetesPedigreeFunction}
                                onChange={handleChange}
                                required
                                tooltip="Diabetes pedigree function based on family history."
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

export default WomanDiabetes;
