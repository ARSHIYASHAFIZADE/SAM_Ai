import React from 'react';
import Navbar from './Navbar';
import Footer from './Footer';

interface MainLayoutProps {
    children: React.ReactNode;
    isAuthenticated?: boolean;
    onLogout?: () => void;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children, isAuthenticated = false, onLogout = () => { } }) => {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
            <Navbar isAuthenticated={isAuthenticated} onLogout={onLogout} />
            <main style={{ flex: 1 }}>{children}</main>
            <Footer />
        </div>
    );
};

export default MainLayout;
