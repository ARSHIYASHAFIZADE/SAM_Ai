import React, { useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './utils/api';
import styles from './HeartDisease.module.css';
import FormGrid from './components/common/FormGrid';
import FormField from './components/common/FormField';
import Input from './components/common/Input';
import Select from './components/common/Select';
import SectionTitle from './components/common/SectionTitle';

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
            [e.target.name as keyof typeof inputData]: e.target.value
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
            const response = await axios.post(`${API_BASE_URL}/detect_heart`, { data: numericData });
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
                <h1 style={{ marginBottom: '2rem' }}>Heart Disease Detection</h1>
                <form id="heartDiseaseForm" onSubmit={handleSubmit}>
                    <FormGrid>
                        <SectionTitle title="Demographics" />
                        <FormField>
                            <Input label="Age" type="number" id="Age" name="age" placeholder="Years" value={inputData.age} onChange={handleChange} required tooltip="Patient's age in years." />
                        </FormField>
                        <FormField>
                            <Select label="Sex" id="Sex" name="sex" value={inputData.sex} onChange={handleChange} required options={[{ value: '0', label: 'Female' }, { value: '1', label: 'Male' }]} tooltip="Patient's biological sex." />
                        </FormField>

                        <SectionTitle title="Symptoms" />
                        <FormField>
                            <Select label="Chest Pain Type" id="ChestPain" name="cp" value={inputData.cp} onChange={handleChange} required options={[{ value: '1', label: 'Typical Angina' }, { value: '2', label: 'Atypical Angina' }, { value: '3', label: 'Non-Anginal Pain' }, { value: '4', label: 'Asymptomatic' }]} tooltip="Type of chest pain experienced." />
                        </FormField>
                        <FormField>
                            <Select label="Exercise Induced Angina" id="ExerciseAngina" name="exang" value={inputData.exang} onChange={handleChange} required options={[{ value: '0', label: 'No' }, { value: '1', label: 'Yes' }]} tooltip="Angina induced by exercise?" />
                        </FormField>

                        <SectionTitle title="Vitals & Labs" />
                        <FormField>
                            <Input label="Resting BP" type="number" id="RestingBP" name="trestbps" placeholder="mm Hg" value={inputData.trestbps} onChange={handleChange} required tooltip="Resting blood pressure in mm Hg." />
                        </FormField>
                        <FormField>
                            <Input label="Cholesterol" type="number" id="Cholesterol" name="chol" placeholder="mg/dl" value={inputData.chol} onChange={handleChange} required tooltip="Serum cholesterol in mg/dl." />
                        </FormField>
                        <FormField>
                            <Select label="Fasting BS > 120 mg/dl" id="FastingBS" name="fbs" value={inputData.fbs} onChange={handleChange} required options={[{ value: '0', label: 'No (< 120 mg/dl)' }, { value: '1', label: 'Yes (> 120 mg/dl)' }]} tooltip="Fasting blood sugar > 120 mg/dl." />
                        </FormField>
                        <FormField>
                            <Input label="Max Heart Rate" type="number" id="MaxHR" name="thalach" placeholder="bpm" value={inputData.thalach} onChange={handleChange} required tooltip="Maximum heart rate achieved." />
                        </FormField>

                        <SectionTitle title="Diagnostic Tests & Risk Factors" />
                        <FormField>
                            <Select label="Resting ECG" id="RestingECG" name="restecg" value={inputData.restecg} onChange={handleChange} required options={[{ value: '0', label: 'Normal' }, { value: '1', label: 'Having ST-T Wave Abnormality' }, { value: '2', label: 'Showing Probable or Definite Left Ventricular Hypertrophy' }]} tooltip="Resting electrocardiographic results." />
                        </FormField>
                        <FormField>
                            <Input label="ST Depression (Oldpeak)" type="number" id="Oldpeak" name="oldpeak" placeholder="e.g., 2.3" value={inputData.oldpeak} onChange={handleChange} required tooltip="ST depression induced by exercise relative to rest." />
                        </FormField>
                        <FormField>
                            <Select label="Slope of Peak Exercise ST Segment" id="Slope" name="slope" value={inputData.slope} onChange={handleChange} required options={[{ value: '1', label: 'Upsloping' }, { value: '2', label: 'Flat' }, { value: '3', label: 'Downsloping' }]} tooltip="The slope of the peak exercise ST segment." />
                        </FormField>
                        <FormField>
                            <Input label="Major Vessels Configured" type="number" id="CA" name="ca" placeholder="0 to 3" value={inputData.ca} onChange={handleChange} required tooltip="Number of major vessels (0-3) colored by fluoroscopy." />
                        </FormField>
                        <FormField fullWidth>
                            <Select label="Thalassemia" id="Thal" name="thal" value={inputData.thal} onChange={handleChange} required options={[{ value: '3', label: 'Normal' }, { value: '6', label: 'Fixed Defect' }, { value: '7', label: 'Reversible Defect' }]} tooltip="Thalassemia result." />
                        </FormField>

                        <FormField fullWidth>
                            <button type="submit" className="medical-submit-btn">Submit</button>
                        </FormField>
                    </FormGrid>

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
