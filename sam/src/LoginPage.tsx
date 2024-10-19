import React, { useState } from "react";
import httpClient from "./httpClient";
import styles from "./login.module.css"; 
import {Link} from 'react-router-dom'
interface LoginPageProps {
    onLogin: (userId: string) => void;
}

const LoginPage: React.FC<LoginPageProps> = ({ onLogin }) => {
    const [email, setEmail] = useState<string>("");
    const [password, setPassword] = useState<string>("");

    const logInUser = async () => {
        try {
            const resp = await httpClient.post("https://sam-ai-mu6e.onrender.com/api/login", {
                email,
                password,
            });

            const userId = resp.data.id;
            onLogin(userId);

            window.location.href = "/";
        } catch (error: any) {
            if (error.response && error.response.status === 401) {
                alert("Invalid credentials");
            } else {
                console.error("An unexpected error occurred:", error);
            }
        }finally{
            setEmail(""),
            setPassword("")
        }
    };

    return (
        <div className={styles.container}>
            <div className={styles.form}>
                <h1>Log Into Your Account</h1>
                <form>
                    <div>
                        <label className={styles.label}>Email: </label>
                        <input
                            type="text"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className={styles.input}
                        />
                    </div>
                    <div>
                        <label className={styles.label}>Password: </label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className={styles.input}
                        />
                    </div>
                    <button
                        type="button"
                        onClick={logInUser}
                        className={styles.button}
                    >
                        Submit
                    </button>
                    <h4><Link to="/Register" style={{ textDecoration: 'none', color: 'inherit'}}>don't have an account Sign Up</Link></h4>
                </form>
            </div>
        </div>
    );
};

export default LoginPage;
