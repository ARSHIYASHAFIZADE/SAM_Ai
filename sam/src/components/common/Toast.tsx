import React, { useEffect } from 'react';
import '../../styles/global.css';

interface ToastProps {
    message: string;
    type: 'success' | 'error';
    onClose: () => void;
    duration?: number;
}

const Toast: React.FC<ToastProps> = ({ message, type, onClose, duration = 3000 }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onClose();
        }, duration);

        return () => clearTimeout(timer);
    }, [duration, onClose]);

    return (
        <div className={`toast toast-${type}`}>
            <div className="toast-content">
                <span className="toast-icon">
                    {type === 'success' ? '✓' : '⚠'}
                </span>
                <span className="toast-message">{message}</span>
            </div>
            <button className="toast-close" onClick={onClose}>×</button>

            <style>{`
                .toast {
                    position: fixed;
                    bottom: 2rem;
                    right: 2rem;
                    min-width: 300px;
                    padding: 1rem;
                    border-radius: var(--radius-lg);
                    background: rgba(15, 23, 42, 0.95);
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                    color: white;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    z-index: 1000;
                    animation: slideIn 0.3s ease-out;
                }
                
                .toast-success {
                    border-left: 4px solid var(--success, #10b981);
                }
                
                .toast-error {
                    border-left: 4px solid var(--danger, #ef4444);
                }
                
                .toast-content {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                }
                
                .toast-icon {
                    font-size: 1.2rem;
                    font-weight: bold;
                }
                
                .toast-success .toast-icon { color: var(--success, #10b981); }
                .toast-error .toast-icon { color: var(--danger, #ef4444); }
                
                .toast-close {
                    background: none;
                    border: none;
                    color: rgba(255, 255, 255, 0.5);
                    font-size: 1.5rem;
                    cursor: pointer;
                    padding: 0 0.5rem;
                    line-height: 1;
                    transition: color 0.2s;
                }
                
                .toast-close:hover {
                    color: white;
                }
                
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `}</style>
        </div>
    );
};

export default Toast;
