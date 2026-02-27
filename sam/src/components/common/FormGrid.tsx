import React from 'react';

interface FormGridProps {
    children: React.ReactNode;
    className?: string;
    style?: React.CSSProperties;
}

export const FormGrid: React.FC<FormGridProps> = ({ children, className = '', style }) => {
    return (
        <div style={style} className={`medical-form-grid ${className}`}>
            {children}
        </div>
    );
};

export default FormGrid;
