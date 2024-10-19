import './App.css'; 

const About = () => {
  return (
    <div className="about-page">
      <section className="about-intro">
        <h1>About Artificial Intelligence in Medicine</h1>
        <p>
          Artificial intelligence (AI) in medicine helps diagnose, treat, and prevent diseases with precision and speed, acting as a powerful assistant alongside doctors. Here’s a closer look at its applications and advantages:
        </p>
      </section>

      <section className="ai-applications">
        <h2>Applications of AI in Medicine</h2>
        <ul>
          <li>AI helps doctors easily access medical data and improve patient experiences.</li>
          <li>It facilitates the distribution of personal health records on smartphones and monitors vital functions using biosensors.</li>
          <li>Artificial neural networks can match radiologists in diagnosing conditions like breast cancer.</li>
          <li>In prescribing medicine, AI aids in quicker and more accurate diagnoses.</li>
          <li>It assists in disease prevention through diagnostic data and improved predictions.</li>
        </ul>
      </section>

      <section className="ai-advantages">
        <h2>Advantages of AI in Medicine</h2>
        <ul>
          <li><strong>Faster and Cost-Effective:</strong> Speeds up research and saves hours of productivity for medical institutions.</li>
          <li><strong>Diagnostic Accuracy:</strong> Enhances diagnostic accuracy and allows doctors to focus on critical patient signals.</li>
          <li><strong>Valuable Insights:</strong> Provides insights for healthcare professionals and patients, addressing many human problems.</li>
        </ul>
      </section>

      <section className="additional-info">
        <h2>How SAM AI Utilizes AI</h2>
        <p>
          SAM AI provides access to past tests on a personal dashboard, allowing users to download them again. Additionally, it offers consultations on diseases and other health-related queries.
        </p>
      </section>
    </div>
  );
}

export default About;
