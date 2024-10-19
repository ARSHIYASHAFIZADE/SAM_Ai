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
      <p id="contact-header">Hover over the cards to see members information</p>
      <div className="about-content" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <div
          className="about-card"
          style={{
            width: '300px', // Set a fixed width for the card
            height: '400px', // Set a fixed height for the card
            perspective: '1000px', // Optional for 3D effect on hover
            margin: '20px', // Space around the card
            display: 'flex',
            flexDirection: 'column', // Align content vertically
            justifyContent: 'center', // Center content vertically within the card
            alignItems: 'center', // Center content horizontally within the card
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)', // Optional: add shadow for depth
            overflow: 'hidden', // Optional: prevent overflow of content
          }}
        >
          <div className="card-front" style={{ width: '100%', height: '100%' }}>
            <img src={Arshiya} alt="AI Member" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          </div>
          <div className="card-back" style={{ padding: '16px' }}>
            <h3>Arshiya Shafizade</h3>
            <p>
              <strong>Name:</strong> Arshiya Shafizade<br />
              <hr id='space' />
              <strong>Email:</strong> a683351@gmail.com<br />
              <hr id='space' />
              <strong>Phone:</strong> +60 172821378<br />
              <hr id='space' />
              <strong>Role:</strong> web developer & AI specialist
            </p>
          </div>
        </div>
      </div>
      <div className="contact-info" style={{ marginTop: '-170px' }}> {/* Adjust this value */}
        <h3>Get in Touch</h3>
        <p><FontAwesomeIcon icon={faPhoneAlt} /> +60 172821378</p>
        <p><FontAwesomeIcon icon={faEnvelope} /> a683351@gmail.com</p>
        <p>Follow us on social media!</p>
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
