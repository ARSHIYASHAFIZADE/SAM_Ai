import React from 'react';
import Tooltip from './Tooltip';
import '../../styles/global.css';

interface SelectOption {
    value: string | number;
    label: string;
}

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
    label?: string;
    error?: string;
    options: SelectOption[];
    tooltip?: string;
}

const Select: React.FC<SelectProps> = ({
    label,
    error,
    options,
    tooltip,
    className = '',
    style,
    required,
    ...props
}) => {
    const containerStyle: React.CSSProperties = {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'flex-end',
        height: '100%',
        fontFamily: 'var(--font-main)',
    };

    const labelContainerStyle: React.CSSProperties = {
        display: 'flex',
        alignItems: 'center',
        gap: '0.375rem',
        marginBottom: '0.75rem',
        marginTop: 'auto',
    };

    const labelStyle: React.CSSProperties = {
        color: 'var(--text-secondary)',
        fontSize: 'var(--font-size-sm)',
        fontWeight: 'var(--font-weight-medium)',
        letterSpacing: 'var(--letter-spacing-normal)',
    };

    const requiredIndicatorStyle: React.CSSProperties = {
        color: 'var(--danger)',
        fontSize: 'var(--font-size-sm)',
        marginLeft: '0.125rem',
    };

    const selectStyle: React.CSSProperties = {
        width: '100%',
        height: '48px',
        boxSizing: 'border-box',
        padding: '0 var(--spacing-md)',
        paddingRight: '2.5rem',
        background: 'var(--surface-light)',
        border: error ? '1.5px solid var(--danger)' : '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: 'var(--radius-md)',
        color: 'var(--text-main)',
        fontFamily: 'var(--font-main)',
        fontSize: 'var(--font-size-base)',
        fontWeight: 'var(--font-weight-normal)',
        outline: 'none',
        transition: 'all var(--transition-fast)',
        cursor: 'pointer',
        appearance: 'none',
        WebkitAppearance: 'none',
        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23A8B2BD'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`,
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'right var(--spacing-md) center',
        backgroundSize: '1.25rem',
    };

    const errorStyle: React.CSSProperties = {
        color: 'var(--danger)',
        fontSize: 'var(--font-size-xs)',
        marginTop: 'var(--spacing-xs)',
        lineHeight: 'var(--line-height-normal)',
    };

    const handleFocus = (e: React.FocusEvent<HTMLSelectElement>) => {
        e.target.style.borderColor = 'var(--primary)';
        e.target.style.boxShadow = 'var(--focus-ring)';
    };

    const handleBlur = (e: React.FocusEvent<HTMLSelectElement>) => {
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
            <select
                style={{ ...selectStyle, ...style }}
                className={className}
                onFocus={handleFocus}
                onBlur={handleBlur}
                required={required}
                {...props}
            >
                <option value="" disabled>Select an option</option>
                {options.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                        {opt.label}
                    </option>
                ))}
            </select>
            {error && <span style={errorStyle}>{error}</span>}
        </div>
    );
};

export default Select;
