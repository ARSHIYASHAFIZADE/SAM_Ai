import { Link } from 'react-router-dom'
import sam1 from "./assets/sam-Ai.jpg";
import sam2 from "./assets/sam-ai2.jpg";
import samvid1 from "./assets/sam-vid1.mp4";
import dm from "./assets/dm.png";
import dw from "./assets/dw.png";
import heart from "./assets/h.png";
import liver from"./assets/liver.png"
import bc from "./assets/bc.png"
const Home = () => {
    return (
        <div>
            <section id="home" className="home-section">
                <div className="home-content">
                    <h1>Welcome to SAM AI</h1>
                    <p>Revolutionizing healthcare with cutting-edge AI technology.</p>
                </div>
            </section>
            <section id="features" className="features-section">
                <h2>Features</h2>
                <div className="features-grid">
                    <div className="feature-card">
                        <Link to="/ManDiabetes" style={{ textDecoration: 'none', color: 'inherit' }}>
                            <h3>Man Diabeties</h3>
                            <img src={dm} alt="description" />
                            <p>Get instant and accurate diagnostics using our AI technology.</p>
                        </Link>
                    </div>
                    <div className="feature-card">
                        <Link to="/WomanDiabetes" style={{ textDecoration: 'none', color: 'inherit' }}>
                            <h3>Women Diabeties</h3>
                            <img src={dw} alt="description" />
                            <p>Get instant and accurate diagnostics using our AI technology.</p>
                        </Link>
                    </div>
                    <div className="feature-card">

                        <Link to="/HeartDisease" style={{ textDecoration: 'none', color: 'inherit' }}>
                            <h3>Heart diagnostics</h3>
                            <img src={heart} alt="description" />
                            <p>Get instant and accurate diagnostics using our AI technology.</p>
                        </Link>
                    </div>
                    <div className="feature-card">

                        <Link to="/LiverDetection" style={{ textDecoration: 'none', color: 'inherit' }}>
                            <h3>Liver diagnostics</h3>
                            <img src={liver} alt="description" />
                            <p>Get instant and accurate diagnostics using our AI technology.</p>
                        </Link>
                    </div>
                    <div className="feature-card">

<Link to="/BreastCancerDetection" style={{ textDecoration: 'none', color: 'inherit' }}>
    <h3>BreastCancerDetection</h3>
    <img src={bc} alt="description" />
    <p>Get instant and accurate diagnostics using our AI technology.</p>
</Link>
</div>
                </div>
            </section>
            <h1 id="intro-vid-text">Intro to SAM Ai</h1>
            <section id="features" className="features-video">
                <div className="feature-video-card">
                    <video src={samvid1} id="samvid1" controls />
                </div>
            </section>
            <section id="about" className="about-section">
                <h2>About Us</h2>
                <div className="about-content">
                    <div className="about-card">
                        <div className="card-front">
                            <h3>Our Mission</h3>
                            <p>To improve healthcare outcomes using advanced AI technologies.</p>
                        </div>
                        <div className="card-back">
                            <h3>Vision</h3>
                            <p>We envision a world where AI empowers individuals to lead healthier lives.</p>
                        </div>
                    </div>
                    <div className="about-card">
                        <div className="card-front">
                            <h3>Our Team</h3>
                            <p>A diverse team of experts in AI, healthcare, and technology.</p>
                        </div>
                        <div className="card-back">
                            <h3>Join Us</h3>
                            <p>Be part of our mission to transform healthcare.</p>
                        </div>
                    </div>
                    <div className="about-card">
                        <div className="card-front">
                            <img src={sam1} alt="description" />
                        </div>
                        <div className="card-back">
                            <img src={sam2} alt="description" />
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default Home;