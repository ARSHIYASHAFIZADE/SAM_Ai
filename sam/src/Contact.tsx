import Arshiya from './assets/Arshiya.png'; 
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEnvelope, faPhoneAlt } from '@fortawesome/free-solid-svg-icons';
import { faLinkedinIn, faGithub } from '@fortawesome/free-brands-svg-icons';

const Contact = () => {
  return (
    <div> 
      <div className="about-card" style={{ marginTop: "100px" }}>
        <div className="card-front">
          <img src={Arshiya} alt="AI Member" />
        </div>
        <div className="card-back">
          <h3>Arshiya Shafizade</h3>
          <p>
            <strong>Name:</strong> Arshiya Shafizade<br /><hr />
            <strong>Email:</strong> shafizadearshiya@gmail.com<br /><hr />
            <strong>Phone:</strong> +60 172821378<br /><hr />
            <strong>Role:</strong> Full stack web developer & AI specialist
          </p>
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
