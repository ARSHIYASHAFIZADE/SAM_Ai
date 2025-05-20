import Arshiya from './assets/Arshiya.png'; 
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEnvelope, faPhoneAlt } from '@fortawesome/free-solid-svg-icons';
import { faLinkedinIn, faGithub } from '@fortawesome/free-brands-svg-icons';
const Contact = () => {
  return (
    
    <div>
      <section id="home" className="home-section">
        <div className="home-content">
          <h1>Welcome to SAM AI</h1>
          <p>Revolutionizing healthcare with cutting-edge AI technology.</p>
        </div>
      </section>
      <div className="about-content">
        <div className="about-card">
          <div className="card-front">
            <img src={Arshiya} alt="AI Member" />
          </div>
          <div className="card-back">
            <h3>Arshiya Shafizade</h3>
            <p><strong>Name:</strong> Arshiya Shafizade<br></br><hr></hr>
              <strong>Email:</strong> shafizadearshiya@gmail.com<br></br><hr></hr>
              <strong>Phone:</strong> +60 172821378<br></br><hr></hr>
              <strong>Role:</strong> FUll stack web developer & AI specialist</p>
          </div>
        </div>
      </div>
      <div className="contact-info">
        <h3>Get in Touch</h3>
        <p><FontAwesomeIcon icon={faPhoneAlt} /> +60 172821378</p>
        <p><FontAwesomeIcon icon={faEnvelope} /> shafizadearshiya@gmail.com</p>
        <p>Follow me on social media!</p>
        <div className="social-links">
          <a href="https://www.linkedin.com/in/arshiya-shafizade/" target="_blank" rel="noopener noreferrer">
            <FontAwesomeIcon icon={faLinkedinIn} />
          </a>
          <a href="https://github.com/ARSHIYASHAFIZADE" target="_blank" rel="noopener noreferrer">
            <FontAwesomeIcon icon={faGithub} />
          </a>
        </div>
    </div>
    </div>
  );
}

export default Contact;
