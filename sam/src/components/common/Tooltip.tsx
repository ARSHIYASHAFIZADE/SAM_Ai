import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faInfoCircle } from '@fortawesome/free-solid-svg-icons';

interface TooltipProps {
    text: string;
}

const Tooltip: React.FC<TooltipProps> = ({ text }) => {
    const [isVisible, setIsVisible] = useState(false);

    // Parse three-part tooltip structure if provided
    // Expected format: "What: ... | How: ... | Why: ..."
    const parseTooltip = (text: string) => {
        const parts = text.split('|').map(part => part.trim());
        if (parts.length === 3) {
            return {
                what: parts[0].replace(/^What:\s*/i, '').trim(),
                how: parts[1].replace(/^How:\s*/i, '').trim(),
                why: parts[2].replace(/^Why:\s*/i, '').trim(),
                structured: true
            };
        }
        return { text, structured: false };
    };

    const tooltipContent = parseTooltip(text);

    return (
        <span
            className="tooltip-container"
            onMouseEnter={() => setIsVisible(true)}
            onMouseLeave={() => setIsVisible(false)}
            onFocus={() => setIsVisible(true)}
            onBlur={() => setIsVisible(false)}
            tabIndex={0}
            role="tooltip"
            aria-label={text}
        >
            <span className="tooltip-icon-wrapper">
                <FontAwesomeIcon icon={faInfoCircle} className="tooltip-icon" />
            </span>
            {isVisible && (
                <div className="tooltip-popup" role="tooltip">
                    {tooltipContent.structured ? (
                        <div className="tooltip-structured">
                            <div className="tooltip-section">
                                <span className="tooltip-label">What it is:</span>
                                <span className="tooltip-text">{tooltipContent.what}</span>
                            </div>
                            <div className="tooltip-section">
                                <span className="tooltip-label">How to enter:</span>
                                <span className="tooltip-text">{tooltipContent.how}</span>
                            </div>
                            <div className="tooltip-section">
                                <span className="tooltip-label">Why it matters:</span>
                                <span className="tooltip-text">{tooltipContent.why}</span>
                            </div>
                        </div>
                    ) : (
                        <div className="tooltip-simple">{tooltipContent.text}</div>
                    )}
                </div>
            )}
            <style>{`
                .tooltip-container {
                    position: relative;
                    display: inline-flex;
                    align-items: center;
                    margin-left: 0.5rem;
                    cursor: help;
                    outline: none;
                }
                
                .tooltip-icon-wrapper {
                    display: inline-flex;
                    justify-content: center;
                    align-items: center;
                    color: var(--primary);
                    transition: all var(--transition-fast);
                }

                .tooltip-icon {
                    font-size: 1.1rem; 
                }
                
                .tooltip-container:hover .tooltip-icon-wrapper {
                    color: var(--primary-hover);
                    transform: scale(1.1);
                }
                
                .tooltip-container:focus-visible .tooltip-icon-wrapper {
                     outline: 2px solid var(--primary);
                     border-radius: 50%;
                }
                
                .tooltip-popup {
                    /* Instant display - no animation */
                    position: absolute;
                    bottom: 100%;
                    left: 50%;
                    transform: translateX(-50%); 
                    margin-bottom: 0.5rem;
                    min-width: 280px;
                    max-width: 360px;
                    padding: var(--spacing-md);
                    background: #FFFFFF;
                    color: #1A1A1A;
                    font-size: var(--font-size-sm);
                    line-height: var(--line-height-relaxed);
                    border-radius: var(--radius-md);
                    box-shadow: var(--shadow-lg);
                    z-index: 1000;
                    pointer-events: none;
                    white-space: normal;
                    border: 1px solid rgba(0, 0, 0, 0.08);
                    /* No transitions or animations */
                }
                
                /* Smart positioning - flip to left if near right edge */
                @media (max-width: 640px) {
                    .tooltip-popup {
                        left: auto;
                        right: -10px;
                        transform: none;
                    }
                }
                
                /* Arrow pointing to icon */
                .tooltip-popup::after {
                    content: '';
                    position: absolute;
                    top: 100%;
                    left: 50%;
                    margin-left: -6px;
                    border-width: 6px;
                    border-style: solid;
                    border-color: #FFFFFF transparent transparent transparent;
                    filter: drop-shadow(0 1px 1px rgba(0, 0, 0, 0.05));
                }
                
                /* Three-part structured tooltip */
                .tooltip-structured {
                    display: flex;
                    flex-direction: column;
                    gap: var(--spacing-sm);
                }
                
                .tooltip-section {
                    display: flex;
                    flex-direction: column;
                    gap: 0.25rem;
                }
                
                .tooltip-label {
                    font-weight: var(--font-weight-semibold);
                    font-size: var(--font-size-xs);
                    text-transform: uppercase;
                    letter-spacing: var(--letter-spacing-wide);
                    color: #666;
                }
                
                .tooltip-text {
                    font-size: var(--font-size-sm);
                    color: #1A1A1A;
                    line-height: var(--line-height-relaxed);
                }
                
                /* Simple tooltip (fallback) */
                .tooltip-simple {
                    font-size: var(--font-size-sm);
                    color: #1A1A1A;
                    line-height: var(--line-height-relaxed);
                }
            `}</style>
        </span>
    );
};

export default Tooltip;
