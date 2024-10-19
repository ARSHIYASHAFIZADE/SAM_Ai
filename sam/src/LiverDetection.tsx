import React, { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';
import styles from './LiverDetection.module.css';

interface PredictionResult {
    prediction: number; 
    probability_healthy_liver: string; 
}

const LiverDetection: React.FC = () => {
    const [inputData, setInputData] = useState({
        Age: '',
        Gender: '',
        Total_Bilirubin: '',
        Direct_Bilirubin: '',
        Alkaline_Phosphotase: '',
        Alamine_Aminotransferase: '',
        Aspartate_Aminotransferase: '',
        Total_Proteins: '',
        Albumin: '',
        Albumin_and_Globulin_Ratio: ''
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
            Gender: inputData.Gender === 'Male' ? 0 : 1,  // Convert gender to 0 or 1
            Total_Bilirubin: Number(inputData.Total_Bilirubin),
            Direct_Bilirubin: Number(inputData.Direct_Bilirubin),
            Alkaline_Phosphotase: Number(inputData.Alkaline_Phosphotase),
            Alamine_Aminotransferase: Number(inputData.Alamine_Aminotransferase),
            Aspartate_Aminotransferase: Number(inputData.Aspartate_Aminotransferase),
            Total_Proteins: Number(inputData.Total_Proteins),
            Albumin: Number(inputData.Albumin),
            Albumin_and_Globulin_Ratio: Number(inputData.Albumin_and_Globulin_Ratio)
        };

        const formattedData = [
            numericData.Age,
            numericData.Gender,
            numericData.Total_Bilirubin,
            numericData.Direct_Bilirubin,
            numericData.Alkaline_Phosphotase,
            numericData.Alamine_Aminotransferase,
            numericData.Aspartate_Aminotransferase,
            numericData.Total_Proteins,
            numericData.Albumin,
            numericData.Albumin_and_Globulin_Ratio
        ];

        try {
            const response = await axios.post('hhttps://sam-ai-mu6e.onrender.com/api/detect_liver', {
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
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Total_Bilirubin">Total Bilirubin:</label>
                        <input
                            type="number"
                            id="Total_Bilirubin"
                            name="Total_Bilirubin"
                            placeholder="Total Bilirubin"
                            value={inputData.Total_Bilirubin}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Direct_Bilirubin">Direct Bilirubin:</label>
                        <input
                            type="number"
                            id="Direct_Bilirubin"
                            name="Direct_Bilirubin"
                            placeholder="Direct Bilirubin"
                            value={inputData.Direct_Bilirubin}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Alkaline_Phosphotase">Alkaline Phosphotase:</label>
                        <input
                            type="number"
                            id="Alkaline_Phosphotase"
                            name="Alkaline_Phosphotase"
                            placeholder="Alkaline Phosphotase"
                            value={inputData.Alkaline_Phosphotase}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Alamine_Aminotransferase">Alamine Aminotransferase:</label>
                        <input
                            type="number"
                            id="Alamine_Aminotransferase"
                            name="Alamine_Aminotransferase"
                            placeholder="Alamine Aminotransferase"
                            value={inputData.Alamine_Aminotransferase}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Aspartate_Aminotransferase">Aspartate Aminotransferase:</label>
                        <input
                            type="number"
                            id="Aspartate_Aminotransferase"
                            name="Aspartate_Aminotransferase"
                            placeholder="Aspartate Aminotransferase"
                            value={inputData.Aspartate_Aminotransferase}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Total_Proteins">Total Proteins:</label>
                        <input
                            type="number"
                            id="Total_Proteins"
                            name="Total_Proteins"
                            placeholder="Total Proteins"
                            value={inputData.Total_Proteins}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Albumin">Albumin:</label>
                        <input
                            type="number"
                            id="Albumin"
                            name="Albumin"
                            placeholder="Albumin"
                            value={inputData.Albumin}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="Albumin_and_Globulin_Ratio">Albumin and Globulin Ratio:</label>
                        <input
                            type="number"
                            id="Albumin_and_Globulin_Ratio"
                            name="Albumin_and_Globulin_Ratio"
                            placeholder="Albumin and Globulin Ratio"
                            value={inputData.Albumin_and_Globulin_Ratio}
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
                    <p className={styles.r}>Prediction: {result.prediction === 0 ? 'No liver problems' : 'Liver problems'}</p>
                    <p className={styles.r}>Probability of having a healthy liver: {result.probability_healthy_liver}</p>
                    <button onClick={handleCancel} className={styles.cancelBtn}>Cancel</button>
                </div>
            )}
        </div>
    );
};

export default LiverDetection;
    
