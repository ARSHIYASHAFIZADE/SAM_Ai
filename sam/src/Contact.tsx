import Arshiya from './assets/Arshiya.png'; 
import memberImg from './assets/mem1.jpg';
import member2Img from './assets/mem2.jpg';
import member3Img from './assets/mem3.jpg';
import member4Img from './assets/mem4.jpg';
import member5Img from './assets/mem5.jpg';
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
      <p id="contact-header">Hover over the cards to see members' information</p>
      <div className="about-content">
        <div className="about-card">
          <div className="card-front">
            <img src={Arshiya} alt="AI Member" />
          </div>
          <div className="card-back">
            <h3>Arshiya Shafizade</h3>
            <p><strong>Name:</strong> Arshiya Shafizade<br></br><hr></hr>
              <strong>Email:</strong> a683351.gmail.com<br></br><hr></hr>
              <strong>Phone:</strong> +60 172821378<br></br><hr></hr>
              <strong>Role:</strong> web developer</p>
          </div>
        </div>
        <div className="about-card">
          <div className="card-front">
            <img src={memberImg} alt="Member" />
          </div>
          <div className="card-back">
            <h3>Mani Hajimohammad taghi</h3>
            <p><strong>Name:</strong>  Mani Hajimohammad taghi<br></br><hr></hr>
              <strong>Phone:</strong> +98 9915779169<br></br><hr></hr>
              <strong>Role:</strong> AI Specialist</p>
          </div>
        </div>
        <div className="about-card">
          <div className="card-front">
            <img src={member2Img} alt="Member" />
          </div>
          <div className="card-back">
              <h3>Elena Dadashian</h3>
            <p><strong>Name:</strong>  Elena Dadashian<br></br><hr></hr>
              <strong>Phone:</strong> +98 9377827843<br></br><hr></hr>
              <strong>Role:</strong> AI Specialist</p>
          </div>
        </div>
        <div className="about-card">
          <div className="card-front">
            <img src={member3Img} alt="Member" />
          </div>
          <div className="card-back">
            <h3>Mohammad Nafe karimi</h3>
            <p><strong>Name:</strong>  Mohammad Nafe karimi<br></br><hr></hr>
              <strong>Phone:</strong> +98 9389722299<br></br><hr></hr>
              <strong>Role:</strong> AI Specialist</p>
          </div>
        </div>
        <div className="about-card">
          <div className="card-front">
            <img src={member4Img} alt="Member" />
          </div>
          <div className="card-back">
            <h3>Bitta Tirgar</h3>
            <p><strong>Name:</strong>  Bitta Tirgar<br></br><hr></hr>
              <strong>Phone:</strong> +98 9360153201<br></br><hr></hr>
              <strong>Role:</strong> AI Specialist</p>
          </div>
        </div>
        <div className="about-card">
          <div className="card-front">
            <img src={member5Img} alt="Member" />
          </div>
          <div className="card-back">
            <h3>Meraj Basiri</h3>
            <p><strong>Name:</strong>  Meraj Basiri<br></br><hr></hr>
              <strong>Phone:</strong> +98 9389722299<br></br><hr></hr>
              <strong>Role:</strong> AI Specialist</p>
          </div>
        </div>
      </div>
      <div className="contact-info">
        <h3>Get in Touch</h3>
        <p><FontAwesomeIcon icon={faPhoneAlt} /> +60 172821378</p>
        <p><FontAwesomeIcon icon={faEnvelope} /> contact@samai.com</p>
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
