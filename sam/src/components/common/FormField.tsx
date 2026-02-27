import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import Input from './Input';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faInfoCircle } from '@fortawesome/free-solid-svg-icons';
import './FormField.css';

interface FormFieldProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label: string;
    tooltip?: string;
    error?: string;
    className?: string;
}

export const FormField: React.FC<FormFieldProps> = ({ 
    label, 
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
                top: rect.top, // Base at top of icon, CSS will translate -100% up
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
            <Input 
                {...props} 
                label={undefined} // We handle label internally
                className="medical-input"
                error={error}
            />
        </div>
    );
};

export default FormField;
