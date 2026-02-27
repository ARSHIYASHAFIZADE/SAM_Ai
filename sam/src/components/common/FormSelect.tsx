import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import Select from './Select';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faInfoCircle } from '@fortawesome/free-solid-svg-icons';
import './FormField.css';

interface SelectOption {
    value: string | number;
    label: string;
}

interface FormSelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
    label: string;
    options: SelectOption[];
    tooltip?: string;
    error?: string;
    className?: string;
}

export const FormSelect: React.FC<FormSelectProps> = ({ 
    label, 
    options,
    tooltip, 
    error, 
    className = '', 
    ...props 
}) => {
    const [showTooltip, setShowTooltip] = useState(false);
    const [tooltipStyles, setTooltipStyles] = useState<React.CSSProperties>({});
    const iconRef = useRef<HTMLDivElement>(null);

    const updatePosition = () => {
        if (iconRef.current) {
            const rect = iconRef.current.getBoundingClientRect();
            setTooltipStyles({
                top: rect.top, 
                left: rect.left + rect.width / 2,
            });
        }
    };

    useEffect(() => {
        const handleScroll = () => setShowTooltip(false);
        if (showTooltip) {
            updatePosition();
            window.addEventListener('scroll', handleScroll, true); 
            window.addEventListener('resize', handleScroll);
        }
        return () => {
            window.removeEventListener('scroll', handleScroll, true);
            window.removeEventListener('resize', handleScroll);
        };
    }, [showTooltip]);

    return (
        <div className={`medical-form-field ${className}`}>
            <div className="field-header">
                <label className="field-label">
                    {label}
                    {props.required && <span className="required-mark">*</span>}
                </label>
                {tooltip && (
                    <div 
                        className="info-icon-wrapper"
                        ref={iconRef}
                        onMouseEnter={() => setShowTooltip(true)}
                        onMouseLeave={() => setShowTooltip(false)}
                        onTouchStart={() => setShowTooltip(!showTooltip)}
                    >
                        <FontAwesomeIcon icon={faInfoCircle} className="info-icon" />
                        {showTooltip && createPortal(
                            <div className="medical-tooltip" style={tooltipStyles}>
                                {tooltip}
                            </div>,
                            document.body
                        )}
                    </div>
                )}
            </div>
            <Select 
                {...props} 
                options={options}
                label={undefined} // Handled internally
                className="medical-input"
                error={error}
            />
        </div>
    );
};

export default FormSelect;
