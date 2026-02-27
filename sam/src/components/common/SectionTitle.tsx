import React from 'react';
import FormField from './FormField';

interface SectionTitleProps {
    title: string;
}

export const SectionTitle: React.FC<SectionTitleProps> = ({ title }) => {
    return (
        <FormField fullWidth>
            <h3 style={{
                fontSize: 'var(--font-size-lg)',
                color: 'var(--secondary)',
                marginTop: '1.5rem',
                marginBottom: '0',
                paddingBottom: '0.5rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                width: '100%',
                fontWeight: 600
            }}>
                {title}
            </h3>
        </FormField>
    );
};

export default SectionTitle;
