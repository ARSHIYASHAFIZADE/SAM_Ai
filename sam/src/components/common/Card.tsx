import React from 'react';
import '../../styles/global.css';

interface CardProps {
    children: React.ReactNode;
    className?: string;
    style?: React.CSSProperties;
    hoverable?: boolean;
    onClick?: () => void;
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(({ children, className = '', style, hoverable, onClick }, ref) => {
    const cardStyle: React.CSSProperties = {
        background: 'var(--glass-bg)',
        backdropFilter: 'var(--blur)',
        border: 'var(--glass-border)',
        borderRadius: 'var(--radius-lg)',
        padding: 'var(--spacing-lg)',
        boxShadow: 'var(--shadow-lg)',
        transition: 'transform var(--transition-normal), box-shadow var(--transition-normal)',
        cursor: onClick ? 'pointer' : 'default',
        ...style,
    };

    const handleMouseEnter = (e: React.MouseEvent<HTMLDivElement>) => {
        if (hoverable || onClick) {
            e.currentTarget.style.transform = 'translateY(-5px)';
            e.currentTarget.style.boxShadow = 'var(--shadow-glow)';
            e.currentTarget.style.borderColor = 'var(--primary)';
        }
    };

    const handleMouseLeave = (e: React.MouseEvent<HTMLDivElement>) => {
        if (hoverable || onClick) {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = 'var(--shadow-lg)';
            e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        }
    };

    return (
        <div
            ref={ref}
            style={cardStyle}
            className={`card ${className}`}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onClick={onClick}
        >
            {children}
        </div>
    );
});

export default Card;
