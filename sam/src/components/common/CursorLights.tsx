import React, { useEffect, useState } from 'react';

const CursorLights: React.FC = () => {
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

    useEffect(() => {
        const handleMouseMove = (event: MouseEvent) => {
            setMousePosition({ x: event.clientX, y: event.clientY });
        };

        window.addEventListener('mousemove', handleMouseMove);

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
        };
    }, []);

    const lightStyle = (color: string, size: number, delay: number, opacity: number) => ({
        position: 'fixed' as const,
        top: 0,
        left: 0,
        width: `${size}px`,
        height: `${size}px`,
        backgroundColor: color,
        borderRadius: '50%',
        pointerEvents: 'none' as const,
        transform: `translate(${mousePosition.x - size / 2}px, ${mousePosition.y - size / 2}px)`,
        transition: `transform ${delay}s ease-out`,
        filter: 'blur(100px)',
        opacity: opacity,
        zIndex: 0,
        mixBlendMode: 'screen' as const,
    });

    return (
        <div className="cursor-lights-container" style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 0, overflow: 'hidden' }}>
            {/* Primary Light (Blue) */}
            <div style={lightStyle('var(--primary)', 600, 0.15, 0.15)} />

            {/* Secondary Light (Teal) */}
            <div style={lightStyle('var(--secondary)', 400, 0.25, 0.1)} />

            {/* Accent Light (Purple/Generic) */}
            <div style={lightStyle('#8b5cf6', 300, 0.35, 0.08)} />
        </div>
    );
};

export default CursorLights;
