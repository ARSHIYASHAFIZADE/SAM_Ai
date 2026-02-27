import React, { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './utils/api';
import styles from './LiverDetection.module.css';
import FormGrid from './components/common/FormGrid';
import FormField from './components/common/FormField';
import Input from './components/common/Input';
import Select from './components/common/Select';
import SectionTitle from './components/common/SectionTitle';

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
            [name as keyof typeof inputData]: value
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
                <h1 style={{ marginBottom: '2rem' }}>Liver Health Detection</h1>
                <form id="liverForm" onSubmit={handleSubmit}>
                    <FormGrid>
                        <SectionTitle title="Demographics" />
                        <FormField>
                            <Input label="Age" type="number" id="Age" name="Age" placeholder="Years" value={inputData.Age} onChange={handleChange} required tooltip="Patient's age in years." />
                        </FormField>
                        <FormField>
                            <Select label="Gender" id="Gender" name="Gender" value={inputData.Gender} onChange={handleChange} required options={[{ value: '0', label: 'Male' }, { value: '1', label: 'Female' }]} tooltip="Patient's biological sex." />
                        </FormField>

                        <SectionTitle title="Lifestyle & Measurements" />
                        <FormField>
                            <Input label="BMI (Body Mass Index)" type="number" id="BMI" name="BMI" step="0.01" placeholder="BMI (e.g., 22.5)" value={inputData.BMI} onChange={handleChange} required tooltip="Body Mass Index." />
                        </FormField>
                        <FormField>
                            <Input label="Physical Activity" type="number" id="PhysicalActivity" name="PhysicalActivity" step="0.01" placeholder="Hours/Week" value={inputData.PhysicalActivity} onChange={handleChange} required tooltip="Hours of physical activity per week." />
                        </FormField>
                        <FormField>
                            <Input label="Alcohol Consumption" type="number" id="AlcoholConsumption" name="AlcoholConsumption" step="0.01" placeholder="Units/Week" value={inputData.AlcoholConsumption} onChange={handleChange} required tooltip="Alcohol consumption in units per week." />
                        </FormField>
                        <FormField>
                            <Select label="Smoking History" id="Smoking" name="Smoking" value={inputData.Smoking} onChange={handleChange} required options={[{ value: '1', label: 'Yes' }, { value: '0', label: 'No' }]} tooltip="History of smoking." />
                        </FormField>

                        <SectionTitle title="Medical History" />
                        <FormField>
                            <Select label="Genetic Risk Level" id="GeneticRisk" name="GeneticRisk" value={inputData.GeneticRisk} onChange={handleChange} required options={[{ value: '0', label: 'Low' }, { value: '1', label: 'Medium' }, { value: '2', label: 'High' }]} tooltip="Assessment of genetic risk based on family history." />
                        </FormField>
                        <FormField>
                            <Select label="Diabetes Status" id="Diabetes" name="Diabetes" value={inputData.Diabetes} onChange={handleChange} required options={[{ value: '1', label: 'Yes' }, { value: '0', label: 'No' }]} tooltip="Does the patient have diabetes?" />
                        </FormField>

                        <SectionTitle title="Lab Values" />
                        <FormField>
                            <Input label="Cholesterol Level" type="number" id="Cholesterol" name="Cholesterol" step="0.01" placeholder="mg/dl" value={inputData.Cholesterol} onChange={handleChange} required tooltip="Serum cholesterol levels in mg/dl." />
                        </FormField>
                        <FormField>
                            <Input label="Liver Function Score" type="number" id="LiverFunctionTest" name="LiverFunctionTest" step="0.01" placeholder="e.g., 1.2" value={inputData.LiverFunctionTest} onChange={handleChange} required tooltip="Standardized liver function test score." />
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
                    <p className={styles.r}>Prediction: {result.prediction === 1 ? 'Liver Problem Detected' : 'No Liver Problems'}</p>
                    <p className={styles.r}>Confidence: {result.probability_healthy_liver}%</p>
                    <button onClick={handleCancel} className={styles.cancelBtn}>Cancel</button>
                </div>
            )}
        </div>
    );
};

export default LiverDetection;
