import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';

interface RequireAuthProps {
    children: React.ReactNode;
    isAuthenticated: boolean;
}

const RequireAuth: React.FC<RequireAuthProps> = ({ children, isAuthenticated }) => {
    const location = useLocation();

    if (!isAuthenticated) {
        // Redirect them to the /login page, but save the current location they were
        // trying to go to when they were redirected. This allows us to send them
        // along to that page after they login, which is a nicer user experience.
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    return <>{children}</>;
};

export default RequireAuth;
