import type { AssessmentType } from './assessmentHistory'

export interface RecommendationItem {
  iconName: 'heart' | 'calendar' | 'activity' | 'shield' | 'alert' | 'check' | 'zap' | 'book' | 'target'
  title: string
  body: string
  urgent?: boolean
}

type RiskLevel = 'Low' | 'Moderate' | 'High'

const DATA: Record<AssessmentType, Record<RiskLevel, RecommendationItem[]>> = {
  heart: {
    Low: [
      { iconName: 'check', title: 'Maintain Cardiovascular Health', body: 'Continue regular aerobic exercise of at least 150 minutes per week, a heart healthy diet low in saturated fats, and routine annual check-ups.' },
      { iconName: 'activity', title: 'Monitor Key Markers', body: 'Have blood pressure, cholesterol, and fasting glucose checked annually. Target BP below 120/80 and LDL below 100 mg/dL.' },
      { iconName: 'heart', title: 'Stay Smoke Free', body: 'If you smoke, cessation reduces cardiovascular risk by up to 50 percent within one year. Avoid passive smoke exposure.' },
      { iconName: 'target', title: 'Know Your Numbers', body: 'Track BMI, waist circumference, and blood pressure regularly. Small sustained improvements deliver large long term benefits.' },
    ],
    Moderate: [
      { iconName: 'calendar', title: 'Schedule a Cardiology Review', body: 'Discuss your result with a physician. A formal cardiovascular risk assessment using Framingham or QRISK scoring is recommended within 3 months.' },
      { iconName: 'activity', title: 'Lifestyle Modification Programme', body: 'Structured dietary change and supervised exercise can reduce cardiovascular risk by 30 percent. Ask your doctor about referral.' },
      { iconName: 'shield', title: 'Review Medications', body: 'If you have hypertension, diabetes, or elevated cholesterol, ensure your management plan is current and optimised.' },
      { iconName: 'zap', title: 'Stress Management', body: 'Chronic psychological stress significantly elevates cardiovascular risk. Explore evidence based interventions such as CBT, mindfulness, or structured relaxation.' },
    ],
    High: [
      { iconName: 'alert', title: 'Seek Medical Evaluation — Do Not Delay', body: 'This result indicates patterns consistent with high cardiovascular risk. Consult a cardiologist or your primary care physician promptly.', urgent: true },
      { iconName: 'activity', title: 'Urgent Symptom Monitoring', body: 'Seek emergency care immediately for chest pain, shortness of breath, jaw pain, or left arm discomfort.', urgent: true },
      { iconName: 'calendar', title: 'Diagnostic Testing', body: 'Request an ECG, echocardiogram, and stress test. Ask about coronary CT angiography to rule out significant arterial disease.' },
      { iconName: 'shield', title: 'Risk Factor Control', body: 'Aggressive management of hypertension, cholesterol, and blood sugar is critical. Medication review and possible dose adjustment may be needed.' },
    ],
  },
  liver: {
    Low: [
      { iconName: 'check', title: 'Liver Health Maintenance', body: 'Limit alcohol to under 14 units per week, maintain a healthy weight, and follow a Mediterranean style diet to support hepatic health.' },
      { iconName: 'activity', title: 'Annual Liver Function Test', body: 'Request an LFT panel including ALT, AST, GGT, Bilirubin, and Albumin at your next annual checkup for a baseline record.' },
      { iconName: 'shield', title: 'Vaccination', body: 'Ensure you are vaccinated against Hepatitis A and B, the most preventable causes of liver disease.' },
      { iconName: 'target', title: 'Medication Awareness', body: 'Paracetamol overdose is the leading cause of acute liver failure. Never exceed recommended doses, especially with alcohol.' },
    ],
    Moderate: [
      { iconName: 'calendar', title: 'Gastroenterology Consultation', body: 'Discuss your result with a physician or gastroenterologist. Consider a repeat LFT in 4–6 weeks to trend enzyme levels.' },
      { iconName: 'activity', title: 'Reduce Hepatotoxic Exposures', body: 'Cease alcohol consumption, review all medications including supplements for hepatotoxicity, and avoid unnecessary NSAID use.' },
      { iconName: 'shield', title: 'Hepatitis Screening', body: 'Request hepatitis B surface antigen and hepatitis C antibody tests if not recently screened.' },
      { iconName: 'zap', title: 'Dietary Adjustments', body: 'Reduce sugar, refined carbohydrates, and saturated fat. Nonalcoholic fatty liver disease, often called NAFLD, can be significantly reversed with weight loss of 5–10 percent.' },
    ],
    High: [
      { iconName: 'alert', title: 'Prompt Medical Assessment Required', body: 'Enzyme patterns suggest significant hepatic stress. See a gastroenterologist or hepatologist within 2–4 weeks.', urgent: true },
      { iconName: 'calendar', title: 'Comprehensive Hepatic Workup', body: 'Request an ultrasound of the abdomen, fibroscan for liver stiffness assessment, and a full viral hepatitis screen. FibroTest may also be indicated.' },
      { iconName: 'shield', title: 'Immediate Alcohol Cessation', body: 'If alcohol is involved, complete cessation is essential. Consider referral to an alcohol support service.', urgent: true },
      { iconName: 'activity', title: 'Monitor for Complications', body: 'Report abdominal swelling, yellowing of the eyes or skin, dark urine, or easy bruising to a doctor immediately.' },
    ],
  },
  'diabetes-female': {
    Low: [
      { iconName: 'check', title: 'Preventive Lifestyle Habits', body: 'Maintain a BMI of 18.5–24.9, engage in 150 minutes per week of moderate aerobic activity, and limit high glycaemic index foods.' },
      { iconName: 'activity', title: 'Annual HbA1c Screening', body: 'Request fasting glucose or HbA1c at your annual checkup, especially if you had gestational diabetes in a prior pregnancy.' },
      { iconName: 'target', title: 'Weight Management', body: 'Even a 5 percent reduction in body weight significantly lowers diabetes risk. Focus on sustainable dietary changes rather than short term diets.' },
      { iconName: 'shield', title: 'Know Gestational Diabetes History', body: 'Women with prior gestational diabetes have a 7 to 10 times higher lifetime risk of Type 2 diabetes. Long term monitoring is essential.' },
    ],
    Moderate: [
      { iconName: 'calendar', title: 'Endocrinology or GP Review', body: 'Discuss your risk profile with a physician. Prediabetes, indicated by an HbA1c reading of 5.7 to 6.4 percent, may be present and requires active management.' },
      { iconName: 'activity', title: 'Structured Diabetes Prevention Programme', body: 'Evidence based programmes like the NHS DPP reduce diabetes onset by 58 percent. Ask your GP about referral.' },
      { iconName: 'zap', title: 'Glucose Monitoring', body: 'Consider home fasting glucose monitoring if your physician recommends it. Target fasting glucose below 100 mg/dL.' },
      { iconName: 'shield', title: 'Hormonal Health Review', body: 'PCOS is associated with insulin resistance and higher diabetes risk. Discuss hormonal health with your gynaecologist or endocrinologist.' },
    ],
    High: [
      { iconName: 'alert', title: 'Formal Diabetes Evaluation Recommended', body: 'This result indicates patterns strongly associated with diabetes. Request a fasting plasma glucose and HbA1c test promptly.', urgent: true },
      { iconName: 'calendar', title: 'Endocrinologist Referral', body: 'A specialist can assess insulin resistance, screen for complications, and discuss pharmacological options if warranted.' },
      { iconName: 'activity', title: 'Symptom Awareness', body: 'Report excessive thirst, frequent urination, blurred vision, or poor wound healing to your doctor immediately.', urgent: true },
      { iconName: 'shield', title: 'Complication Screening', body: 'If diabetes is confirmed, request retinal screening, kidney function tests including eGFR and uACR, and an annual foot examination.' },
    ],
  },
  'diabetes-male': {
    Low: [
      { iconName: 'check', title: 'Maintain Healthy Metabolic Habits', body: 'Regular physical activity, a balanced diet, and weight maintenance remain the most effective diabetes prevention strategies.' },
      { iconName: 'activity', title: 'Annual Fasting Glucose Check', body: 'Request a fasting blood glucose or HbA1c at your annual health review. Early detection enables effective intervention.' },
      { iconName: 'target', title: 'Reduce Sedentary Time', body: 'Prolonged sitting is an independent risk factor for insulin resistance. Aim for movement breaks every 60 minutes.' },
      { iconName: 'shield', title: 'Monitor Symptoms', body: 'Be alert to persistent thirst, frequent urination, unexplained fatigue, or slow healing wounds — early warning signs of diabetes.' },
    ],
    Moderate: [
      { iconName: 'calendar', title: 'Schedule a GP Review', body: 'Your symptom profile suggests a moderate risk. Request formal diagnostic testing: fasting glucose, HbA1c, and oral glucose tolerance test.' },
      { iconName: 'activity', title: 'Structured Exercise Intervention', body: 'A combination of aerobic and resistance training is most effective at improving insulin sensitivity. Target 150 minutes per week.' },
      { iconName: 'zap', title: 'Dietary Glycaemic Control', body: 'Reduce ultra processed foods, sugary drinks, and refined carbohydrates. Increase fibre, lean protein, and healthy fats.' },
      { iconName: 'shield', title: 'Stress and Sleep', body: 'Chronic sleep deprivation and stress impair insulin signalling. Target 7–9 hours of quality sleep per night.' },
    ],
    High: [
      { iconName: 'alert', title: 'Medical Consultation Strongly Recommended', body: 'Your symptom pattern is highly consistent with diabetes. Do not delay — see a physician for diagnostic blood tests this week.', urgent: true },
      { iconName: 'calendar', title: 'Urgent Diagnostic Workup', body: 'Fasting plasma glucose, HbA1c, C peptide, and ketone testing can quickly confirm diagnosis and guide treatment.' },
      { iconName: 'activity', title: 'Critical Symptom Watch', body: 'Seek emergency care for nausea, vomiting, rapid breathing, or altered consciousness — these may indicate diabetic ketoacidosis.', urgent: true },
      { iconName: 'shield', title: 'Complication Baseline Screening', body: 'Upon diagnosis, request neuropathy assessment, retinal exam, kidney function tests, and lipid panel as a baseline.' },
    ],
  },
  'breast-cancer': {
    Low: [
      { iconName: 'check', title: 'Routine Breast Health Monitoring', body: 'Perform monthly breast self examination and attend all scheduled mammography screenings per national guidelines, typically recommended between ages 40 and 74, every 2 years.' },
      { iconName: 'activity', title: 'Lifestyle Risk Reduction', body: 'Maintaining a healthy weight, limiting alcohol, regular exercise, and avoiding postmenopausal HRT are evidence based strategies to reduce breast cancer risk.' },
      { iconName: 'shield', title: 'Family History Awareness', body: 'If a first degree relative has had breast or ovarian cancer, discuss genetic counselling and BRCA testing with your GP.' },
      { iconName: 'book', title: 'Understand the Result', body: 'A low risk result from SAM AI reflects FNA cytology patterns only — it does not replace clinical examination, mammography, or biopsy.' },
    ],
    Moderate: [
      { iconName: 'calendar', title: 'Clinical Breast Examination', body: 'Request a formal clinical breast examination by a physician. If a lump or abnormality is present, referral for imaging is essential.' },
      { iconName: 'activity', title: 'Diagnostic Imaging', body: 'Ask your doctor about mammography and breast ultrasound. Digital breast tomosynthesis, also called 3D mammography, offers improved detection rates.' },
      { iconName: 'shield', title: 'Enhanced Surveillance', body: 'Depending on your risk profile, your physician may recommend more frequent screening intervals or MRI as a supplemental tool.' },
      { iconName: 'zap', title: 'Know the Warning Signs', body: 'Report any new lump, change in breast size or shape, nipple discharge, skin dimpling, or persistent pain to a doctor promptly.' },
    ],
    High: [
      { iconName: 'alert', title: 'Urgent Specialist Referral Required', body: 'Cytology patterns are consistent with malignant classification. An urgent referral to a breast specialist and formal biopsy are strongly indicated.', urgent: true },
      { iconName: 'calendar', title: 'Core Needle Biopsy', body: 'This is the definitive diagnostic procedure. Combined with triple assessment including clinical examination, imaging, and tissue biopsy, it provides a conclusive diagnosis.', urgent: true },
      { iconName: 'activity', title: 'Comprehensive Breast Imaging', body: 'Request urgent bilateral mammography and ultrasound. Contrast enhanced MRI may be indicated depending on findings.' },
      { iconName: 'shield', title: 'Do Not Delay', body: 'Breast cancer treated at early stages has >90% five year survival. Early diagnosis is the single most important determinant of outcome.', urgent: true },
    ],
  },
}

export function getRecommendations(type: AssessmentType, risk_level: string): RecommendationItem[] {
  const level = risk_level as RiskLevel
  return DATA[type]?.[level] ?? []
}
