import React, { useEffect, useRef } from 'react';

const GooeyBackground: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let width = window.innerWidth;
        let height = window.innerHeight;
        let particles: Particle[] = [];
        const particleCount = 20; // "many" balls
        const mouse = { x: -1000, y: -1000 };

        canvas.width = width;
        canvas.height = height;

        class Particle {
            x: number;
            y: number;
            vx: number;
            vy: number;
            size: number;
            baseX: number;
            baseY: number;
            color: string;

            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 1.5; // Floating speed
                this.vy = (Math.random() - 0.5) * 1.5;
                this.size = Math.random() * 30 + 20; // Varied sizes
                this.baseX = this.x;
                this.baseY = this.y;
                // Mix of main brand colors with LOWER OPACITY
                const colors = ['rgba(59, 130, 246, 0.3)', 'rgba(20, 184, 166, 0.3)', 'rgba(139, 92, 246, 0.3)'];
                this.color = colors[Math.floor(Math.random() * colors.length)];
            }

            update() {
                // Floating movement
                this.x += this.vx;
                this.y += this.vy;

                // Bounce off edges
                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;

                // Mouse interaction (Attraction/Merging)
                const dx = mouse.x - this.x;
                const dy = mouse.y - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const maxDistance = 200; // Range to start merging

                if (distance < maxDistance) {
                    const forceDirectionX = dx / distance;
                    const forceDirectionY = dy / distance;
                    const force = (maxDistance - distance) / maxDistance;

                    // Move towards mouse ('merge')
                    // The closer it is, the stronger the pull
                    this.x += forceDirectionX * force * 3;
                    this.y += forceDirectionY * force * 3;
                }
            }

            draw() {
                if (!ctx) return;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = this.color;
                ctx.fill();
            }
        }

        const init = () => {
            particles = [];
            for (let i = 0; i < particleCount; i++) {
                particles.push(new Particle());
            }
        };

        const animate = () => {
            if (!ctx) return;
            // Clear rect with slight transparency for potential trails, but standard clear is better for goo
            ctx.clearRect(0, 0, width, height);

            // Draw the "Cursor" ball with lower opacity
            ctx.beginPath();
            ctx.arc(mouse.x, mouse.y, 40, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(59, 130, 246, 0.4)'; // Reduced opacity
            ctx.fill();

            particles.forEach(particle => {
                particle.update();
                particle.draw();
            });

            requestAnimationFrame(animate);
        };

        const handleResize = () => {
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width;
            canvas.height = height;
            init();
        };

        const handleMouseMove = (e: MouseEvent) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        };

        window.addEventListener('resize', handleResize);
        window.addEventListener('mousemove', handleMouseMove);

        init();
        animate();

        return () => {
            window.removeEventListener('resize', handleResize);
            window.removeEventListener('mousemove', handleMouseMove);
        };
    }, []);

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            overflow: 'hidden',
            pointerEvents: 'none',
            zIndex: -1, // Use -1 to ensure it is behind everything
            filter: 'url("#goo")',
            opacity: 0.7, // Global opacity reduction for the background
        }}>
            <canvas ref={canvasRef} style={{ display: 'block' }} />

            {/* SVG Filter for the Gooey Effect */}
            <svg style={{ position: 'absolute', width: 0, height: 0 }}>
                <defs>
                    <filter id="goo">
                        {/* Blur the input - increased for softer look */}
                        <feGaussianBlur in="SourceGraphic" stdDeviation="15" result="blur" />
                        {/* High contrast alpha adjustment to threshold the blur */}
                        <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7" result="goo" />
                        <feBlend in="SourceGraphic" in2="goo" />
                    </filter>
                </defs>
            </svg>
        </div>
    );
};

export default GooeyBackground;
