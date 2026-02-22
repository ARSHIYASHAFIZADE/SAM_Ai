import React from 'react';
import Tooltip from './Tooltip';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
    icon?: React.ReactNode;
    tooltip?: string;
}

const Input: React.FC<InputProps> = ({
    label,
    error,
    icon,
    tooltip,
    style,
    className = '',
    required,
    ...props
}) => {
    const containerStyle: React.CSSProperties = {
        display: 'flex',
        flexDirection: 'column',
        marginBottom: 'var(--spacing-md)',
        fontFamily: 'var(--font-main)',
    };

    const labelContainerStyle: React.CSSProperties = {
        display: 'flex',
        alignItems: 'center',
        gap: '0.375rem',
        marginBottom: 'var(--spacing-sm)',
    };

    const labelStyle: React.CSSProperties = {
        fontSize: 'var(--font-size-sm)',
        fontWeight: 'var(--font-weight-medium)',
        color: 'var(--text-secondary)',
        letterSpacing: 'var(--letter-spacing-normal)',
    };

    const requiredIndicatorStyle: React.CSSProperties = {
        color: 'var(--danger)',
        fontSize: 'var(--font-size-sm)',
        marginLeft: '0.125rem',
    };

    const inputWrapperStyle: React.CSSProperties = {
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
    };

    const inputStyle: React.CSSProperties = {
        width: '100%',
        padding: 'var(--spacing-sm) var(--spacing-md)',
        paddingLeft: icon ? '2.75rem' : 'var(--spacing-md)',
        background: 'var(--surface-light)',
        border: error ? '1.5px solid var(--danger)' : '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: 'var(--radius-md)',
        color: 'var(--text-main)',
        fontFamily: 'var(--font-main)',
        fontSize: 'var(--font-size-base)',
        fontWeight: 'var(--font-weight-normal)',
        outline: 'none',
        transition: 'all var(--transition-fast)',
    };

    const iconStyle: React.CSSProperties = {
        position: 'absolute',
        left: 'var(--spacing-md)',
        color: 'var(--text-muted)',
        pointerEvents: 'none',
    };

    const errorStyle: React.CSSProperties = {
        color: 'var(--danger)',
        fontSize: 'var(--font-size-xs)',
        marginTop: 'var(--spacing-xs)',
        lineHeight: 'var(--line-height-normal)',
    };

    const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
        e.target.style.borderColor = 'var(--primary)';
        e.target.style.boxShadow = 'var(--focus-ring)';
    };

    const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
        e.target.style.borderColor = error ? 'var(--danger)' : 'rgba(255, 255, 255, 0.1)';
        e.target.style.boxShadow = 'none';
    };

    return (
        <div style={containerStyle}>
            {label && (
                <div style={labelContainerStyle}>
                    <label style={labelStyle}>
                        {label}
                        {required && <span style={requiredIndicatorStyle}>*</span>}
                    </label>
                    {tooltip && <Tooltip text={tooltip} />}
                </div>
            )}
            <div style={inputWrapperStyle}>
                {icon && <span style={iconStyle}>{icon}</span>}
                <input
                    style={{ ...inputStyle, ...style }}
                    className={className}
                    onFocus={handleFocus}
                    onBlur={handleBlur}
                    required={required}
                    {...props}
                />
            </div>
            {error && <span style={errorStyle}>{error}</span>}
        </div>
    );
};

export default Input;
