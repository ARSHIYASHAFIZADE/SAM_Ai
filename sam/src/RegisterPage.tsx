import React, { useState } from "react";
import httpClient from "./httpClient";
import styles from "./login.module.css"; 
import {Link} from 'react-router-dom'
const RegisterPage: React.FC = () => {
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");

  const registerUser = async () => {
    try {
      await httpClient.post("//localhost:5000/register", {
        email,
        password,
      });

      window.location.href = "/login";
    } catch (error: any) {
      if (error.response.status === 401) {
        alert("Invalid credentials");
      }
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.form}>
        <h1>Create an account</h1>
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
            onClick={() => registerUser()}
            className={styles.button}
          >
            Submit
          </button>
          <h4><Link to="/Login" style={{ textDecoration: 'none', color: 'inherit'}}>have an account? Login</Link></h4>
        </form>
      </div>
    </div>
  );
};

export default RegisterPage;
