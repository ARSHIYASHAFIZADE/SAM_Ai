import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Button from '../common/Button';
import '../../styles/global.css';

interface NavbarProps {
    isAuthenticated: boolean;
    onLogout: () => void;
}

const Navbar: React.FC<NavbarProps> = ({ isAuthenticated, onLogout }) => {
    const navigate = useNavigate();


    const navStyle: React.CSSProperties = {
        position: 'sticky',
        top: 0,
        zIndex: 100,
        background: 'rgba(15, 23, 42, 0.8)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        padding: '1rem 0',
    };

    const containerStyle: React.CSSProperties = {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '0 1rem',
    };

    const brandStyle: React.CSSProperties = {
        fontSize: '1.5rem',
        fontWeight: 700,
        background: 'linear-gradient(135deg, var(--primary), var(--secondary))',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        textDecoration: 'none',
        letterSpacing: '-0.02em',
    };

    const linksStyle: React.CSSProperties = {
        display: 'flex',
        gap: '2rem',
        alignItems: 'center',
        listStyle: 'none',
        margin: 0,
        padding: 0,
    };



    const linkItemStyle: React.CSSProperties = {
        color: 'var(--text-muted)',
        fontSize: '0.95rem',
        fontWeight: 500,
        transition: 'color 0.2s',
        cursor: 'pointer',
    };

    return (
        <nav style={navStyle}>
            <div style={containerStyle}>
                <Link to="/" style={brandStyle}>
                    SAM AI
                </Link>

                {/* Desktop Links */}
                <ul className="nav-links" style={linksStyle}>
                    <li><Link to="/" style={linkItemStyle} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--primary)'} onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}>Home</Link></li>
                    <li><Link to="/about" style={linkItemStyle} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--primary)'} onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}>About</Link></li>

                    {!isAuthenticated ? (
                        <>
                            <li>
                                <Link to="/login">
                                    <Button variant="ghost" size="sm">Log In</Button>
                                </Link>
                            </li>
                            <li>
                                <Link to="/register">
                                    <Button variant="primary" size="sm">Get Started</Button>
                                </Link>
                            </li>
                        </>
                    ) : (
                        <li>
                            <Button variant="secondary" size="sm" onClick={() => {
                                onLogout();
                                navigate('/');
                            }}>Logout</Button>
                        </li>
                    )}
                </ul>
            </div>
        </nav>
    );
};

export default Navbar;
