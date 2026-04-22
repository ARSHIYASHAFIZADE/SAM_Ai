import type { AssessmentType } from './assessmentHistory'

export interface TooltipData { description: string; range: string; relevance: string }

type TooltipMap = Record<string, TooltipData>

const heart: TooltipMap = {
  age:       { description: 'Patient age in years.', range: 'Typical range: 29–77 years', relevance: 'Cardiovascular risk increases significantly after age 45.' },
  sex:       { description: 'Biological sex at birth.', range: 'Select Male or Female from the dropdown', relevance: 'Males have higher baseline cardiovascular risk.' },
  cp:        { description: 'Chest pain type experienced.', range: 'Select the type that best matches your symptoms from the dropdown', relevance: 'Typical angina is a stronger predictor of disease.' },
  trestbps:  { description: 'Resting blood pressure on admission to hospital.', range: 'Normal: 90–120 mmHg', relevance: 'Hypertension above 140 is a major cardiovascular risk factor.' },
  chol:      { description: 'Serum cholesterol level.', range: 'Desirable: <200 mg/dL. High: >240 mg/dL', relevance: 'Elevated cholesterol contributes to arterial plaque formation.' },
  fbs:       { description: 'Whether your fasting blood sugar exceeds 120 mg/dL.', range: 'Select Yes if your last fasting glucose reading was above 120 mg/dL', relevance: 'Elevated fasting glucose is a diabetes indicator and cardiac risk marker.' },
  restecg:   { description: 'Resting ECG results.', range: 'Select the result reported on your ECG from the dropdown', relevance: 'ST and T wave abnormalities indicate prior cardiac stress.' },
  thalach:   { description: 'Maximum heart rate achieved during stress test.', range: 'Target max: 220 minus age. Average: 100–200 bpm', relevance: 'Lower max heart rate during exercise may indicate cardiac limitation.' },
  exang:     { description: 'Angina experienced during physical exertion.', range: 'Select Yes if you experience chest pain during exercise', relevance: 'Exercise induced angina strongly correlates with coronary disease.' },
  oldpeak:   { description: 'ST depression induced by exercise relative to rest.', range: '0.0–6.2 mm. Normal: 0', relevance: 'Greater ST depression suggests more severe ischaemia.' },
  slope:     { description: 'Slope of the peak exercise ST segment.', range: 'Select the ST slope result from your stress test report', relevance: 'Downsloping ST segments suggest significant coronary disease.' },
  ca:        { description: 'Number of major vessels coloured by fluoroscopy.', range: '0–4 vessels', relevance: 'More vessels affected indicates higher severity of coronary artery disease.' },
  thal:      { description: 'Thalassemia blood disorder type.', range: 'Select the result from your nuclear stress test from the dropdown', relevance: 'Reversible defects indicate areas of myocardial ischaemia.' },
}

const liver: TooltipMap = {
  Age:                          { description: 'Patient age in years.', range: '4–90 years', relevance: 'Liver disease risk increases with age.' },
  Total_Bilirubin:              { description: 'Total bilirubin in serum, measuring the liver\'s ability to process metabolic waste.', range: 'Normal: 0.1–1.2 mg/dL', relevance: 'Elevated bilirubin indicates liver cell damage or bile duct obstruction.' },
  Direct_Bilirubin:             { description: 'Conjugated bilirubin processed directly by the liver.', range: 'Normal: <0.3 mg/dL', relevance: 'High direct bilirubin points specifically to hepatocellular damage.' },
  Alkaline_Phosphotase:         { description: 'Enzyme found in liver, bone, and bile ducts.', range: 'Normal: 44–147 IU/L', relevance: 'Elevated levels indicate bile duct obstruction or liver inflammation.' },
  Alamine_Aminotransferase:     { description: 'ALT — a liver specific enzyme released when hepatocytes are damaged.', range: 'Normal: 7–56 IU/L', relevance: 'High ALT is a primary marker of acute liver injury or hepatitis.' },
  Aspartate_Aminotransferase:   { description: 'AST — enzyme found in liver and other tissues throughout the body.', range: 'Normal: 10–40 IU/L', relevance: 'The AST to ALT ratio helps differentiate types of liver disease.' },
  Total_Protiens:               { description: 'Total serum proteins, measuring the liver\'s synthetic function.', range: 'Normal: 6.3–8.2 g/dL', relevance: 'Low total protein may indicate impaired liver protein synthesis.' },
  Albumin:                      { description: 'Primary protein synthesised by the liver.', range: 'Normal: 3.5–5.0 g/dL', relevance: 'Low albumin levels reflect chronic liver disease or malnutrition.' },
  Albumin_and_Globulin_Ratio:   { description: 'Ratio of albumin to globulin proteins in serum.', range: 'Normal: >1.0', relevance: 'A reversed albumin to globulin ratio suggests chronic liver disease or immune disorders.' },
  Gender_Male:                  { description: 'Biological sex.', range: 'Select Male or Female from the dropdown', relevance: 'Males have higher prevalence of alcoholic liver disease.' },
}

const diabetesFemale: TooltipMap = {
  Pregnancies:               { description: 'Number of times the patient has been pregnant.', range: '0–17', relevance: 'Gestational diabetes in prior pregnancies raises future diabetes risk.' },
  Glucose:                   { description: '2-hour plasma glucose from oral glucose tolerance test.', range: 'Normal: <140 mg/dL. Diabetic: >200 mg/dL', relevance: 'Glucose tolerance is the primary marker for diabetes screening.' },
  BloodPressure:             { description: 'Diastolic blood pressure measurement.', range: 'Normal: 60–80 mmHg', relevance: 'Hypertension and diabetes are strongly co-associated risk factors.' },
  SkinThickness:             { description: 'Triceps skin fold thickness used as a proxy for body fat.', range: 'Normal: 10–50 mm', relevance: 'Greater skin fold thickness correlates with insulin resistance.' },
  Insulin:                   { description: '2-hour serum insulin level.', range: 'Normal: 16–166 μU/mL', relevance: 'Elevated insulin indicates insulin resistance, a hallmark of Type 2 diabetes.' },
  BMI:                       { description: 'Body Mass Index, calculated as weight in kg divided by height in metres squared.', range: 'Normal: 18.5–24.9. Obese: ≥30', relevance: 'Obesity is the single strongest modifiable risk factor for Type 2 diabetes.' },
  DiabetesPedigreeFunction:  { description: 'A function scoring diabetes likelihood based on family history.', range: '0.078–2.42, where higher values indicate a stronger genetic link', relevance: 'Captures the hereditary component of diabetes risk.' },
  Age:                       { description: 'Patient age in years.', range: 'Study range: 21–81 years', relevance: 'Type 2 diabetes risk increases with age, especially after 45.' },
}

const diabetesMale: TooltipMap = {
  Age:               { description: 'Patient age in years.', range: '20–65 years', relevance: 'Age is a significant risk factor for Type 2 diabetes in males.' },
  Gender:            { description: 'Biological sex.', range: 'Select Male or Female from the dropdown', relevance: 'Biological sex affects hormonal risk factors for diabetes.' },
  Polyuria:          { description: 'Excessive urination producing more than 2.5 litres of urine per day.', range: 'Select Yes if you currently or frequently experience this', relevance: 'A hallmark symptom of uncontrolled diabetes due to glucose spilling into urine.' },
  Polydipsia:        { description: 'Abnormal excessive thirst.', range: 'Select Yes if you currently or frequently experience this', relevance: 'Follows polyuria as the body tries to compensate for fluid loss.' },
  sudden_weight_loss:{ description: 'Unexplained rapid weight loss.', range: 'Select Yes if you currently or frequently experience this', relevance: 'Indicates the body is breaking down fat and muscle when unable to use glucose.' },
  weakness:          { description: 'General physical weakness or fatigue.', range: 'Select Yes if you currently or frequently experience this', relevance: 'Energy deficit from poor glucose utilisation causes persistent fatigue.' },
  Polyphagia:        { description: 'Excessive hunger even after eating.', range: 'Select Yes if you currently or frequently experience this', relevance: 'Cells starved of glucose signal constant hunger despite eating.' },
  Genital_thrush:    { description: 'Recurrent genital yeast infections.', range: 'Select Yes if you currently or frequently experience this', relevance: 'High blood sugar creates an environment conducive to fungal growth.' },
  visual_blurring:   { description: 'Blurred or fluctuating vision.', range: 'Select Yes if you currently or frequently experience this', relevance: 'High glucose causes the eye lens to swell, altering focal length.' },
  Itching:           { description: 'Generalised or localised skin itching.', range: 'Select Yes if you currently or frequently experience this', relevance: 'Poor circulation and nerve damage from diabetes cause chronic skin irritation.' },
  Irritability:      { description: 'Mood changes or increased irritability.', range: 'Select Yes if you currently or frequently experience this', relevance: 'Blood sugar fluctuations directly affect mood and cognitive function.' },
  delayed_healing:   { description: 'Wounds or cuts take longer than usual to heal.', range: 'Select Yes if you currently or frequently experience this', relevance: 'High glucose impairs immune function and microcirculation needed for healing.' },
  partial_paresis:   { description: 'Partial muscle weakness or paralysis.', range: 'Select Yes if you currently or frequently experience this', relevance: 'A sign of diabetic neuropathy affecting motor nerve function.' },
  muscle_stiffness:  { description: 'Stiffness or tightness in muscles.', range: 'Select Yes if you currently or frequently experience this', relevance: 'Nerve and vascular damage from diabetes causes musculoskeletal symptoms.' },
  Alopecia:          { description: 'Hair loss or thinning.', range: 'Select Yes if you currently or frequently experience this', relevance: 'Vascular and hormonal effects of diabetes can disrupt hair follicle health.' },
  Obesity:           { description: 'Clinical obesity with a BMI of 30 or above.', range: 'Select Yes if your BMI is 30 or above', relevance: 'The strongest modifiable risk factor for Type 2 diabetes.' },
}

const breastCancer: TooltipMap = {
  mean_radius:             { description: 'Mean radius of cell nuclei, measured as distance from centre to perimeter.', range: 'Benign: ~12 μm | Malignant: ~17 μm', relevance: 'Larger nuclei are characteristic of malignant transformation.' },
  mean_texture:            { description: 'Standard deviation of greyscale values, measuring nuclear texture variation.', range: 'Benign: ~17 | Malignant: ~21', relevance: 'Coarser nuclear texture is a hallmark of malignancy.' },
  mean_perimeter:          { description: 'Mean perimeter of cell nuclei.', range: 'Benign: ~78 μm | Malignant: ~115 μm', relevance: 'Closely correlated with radius; larger perimeter means a larger nucleus.' },
  mean_area:               { description: 'Mean area of cell nuclei in the FNA image.', range: 'Benign: ~463 μm² | Malignant: ~978 μm²', relevance: 'Malignant cells have significantly larger nuclear areas.' },
  mean_smoothness:         { description: 'Mean local variation in nuclear radius lengths.', range: '0.07–0.16, where lower values indicate smoother boundaries', relevance: 'More irregular cell borders are associated with malignancy.' },
  mean_compactness:        { description: 'Mean compactness, computed as perimeter squared divided by area minus 1.', range: '0.02–0.35', relevance: 'Higher compactness reflects irregular nuclear shapes.' },
  mean_concavity:          { description: 'Mean severity of concave portions of the nuclear contour.', range: '0–0.43', relevance: 'Deep concavities in the nuclear outline suggest malignant morphology.' },
  mean_concave_points:     { description: 'Mean number of concave portions of the nuclear contour.', range: '0–0.20', relevance: 'Count of irregular indentations — higher in malignant cells.' },
  mean_symmetry:           { description: 'Mean symmetry of cell nuclei.', range: '0.14–0.30', relevance: 'Asymmetric nuclei are a common feature of cancerous cells.' },
  mean_fractal_dimension:  { description: 'Mean coastline approximation of nuclear boundary complexity.', range: '0.05–0.10', relevance: 'Higher fractal dimension indicates more irregular nuclear boundaries.' },
}

const TOOLTIPS: Record<AssessmentType, TooltipMap> = {
  heart: heart,
  liver: liver,
  'diabetes-female': diabetesFemale,
  'diabetes-male': diabetesMale,
  'breast-cancer': breastCancer,
}

export function getTooltip(type: AssessmentType, fieldKey: string): TooltipData | null {
  return TOOLTIPS[type]?.[fieldKey] ?? null
}
