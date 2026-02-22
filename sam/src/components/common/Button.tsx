import React from 'react';
import '../../styles/global.css';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
    size?: 'sm' | 'md' | 'lg';
    isLoading?: boolean;
    leftIcon?: React.ReactNode;
    rightIcon?: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({
    children,
    variant = 'primary',
    size = 'md',
    isLoading,
    leftIcon,
    rightIcon,
    className = '',
    disabled,
    style: propStyle, // Extract style from props to merge manually
    ...props
}) => {
    const baseStyles = {
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '0.5rem',
        borderRadius: 'var(--radius-md)',
        fontWeight: 600,
        cursor: 'pointer',
        transition: 'all var(--transition-fast)',
        border: 'none',
        outline: 'none',
        fontFamily: 'var(--font-main)',
    };

    const variantStyles = {
        primary: {
            background: 'linear-gradient(135deg, var(--primary), var(--primary-dark))',
            color: 'white',
            boxShadow: 'var(--shadow-md)',
        },
        secondary: {
            background: 'transparent',
            border: '1px solid var(--primary)',
            color: 'var(--primary)',
        },
        ghost: {
            background: 'transparent',
            color: 'var(--text-muted)',
        },
        danger: {
            background: 'var(--danger)',
            color: 'white',
        },
    };

    const sizeStyles = {
        sm: { padding: '0.25rem 0.75rem', fontSize: '0.875rem' },
        md: { padding: '0.5rem 1rem', fontSize: '1rem' },
        lg: { padding: '0.75rem 1.5rem', fontSize: '1.125rem' },
    };

    // Merge internal styles with prop styles (propStyle takes precedence)
    const combinedStyle = {
        ...baseStyles,
        ...variantStyles[variant],
        ...sizeStyles[size],
        opacity: disabled || isLoading ? 0.7 : 1,
        cursor: disabled || isLoading ? 'not-allowed' : 'pointer',
        ...propStyle,
    };

    return (
        <button
            style={combinedStyle}
            className={`btn ${className}`}
            disabled={disabled || isLoading}
            {...props}
        >
            {isLoading && <span className="spinner">⌛</span>}
            {!isLoading && leftIcon}
            {children}
            {!isLoading && rightIcon}
        </button>
    );
};

export default Button;
