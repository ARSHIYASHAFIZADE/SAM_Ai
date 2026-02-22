import React from 'react';
import '../../styles/global.css';

const Footer: React.FC = () => {
    const footerStyle: React.CSSProperties = {
        background: 'var(--background)',
        borderTop: '1px solid var(--surface-light)',
        padding: '2rem 0',
        marginTop: 'auto',
    };

    const containerStyle: React.CSSProperties = {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1rem',
        color: 'var(--text-muted)',
        fontSize: '0.875rem',
    };

    return (
        <footer style={footerStyle}>
            <div style={containerStyle} className="container">
                <div style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>SAM AI</div>
                <div>&copy; {new Date().getFullYear()} Pioneering Medical Diagnostics.</div>
            </div>
        </footer>
    );
};

export default Footer;
