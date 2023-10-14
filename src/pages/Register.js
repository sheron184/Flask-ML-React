import React, { useEffect, useState } from 'react';
import { Container, Paper, Typography, TextField, Button } from '@mui/material';
import { Person, Email, Lock } from '@mui/icons-material';
import axiosApi from '../axios';
import axios from 'axios';
import { Link } from 'react-router-dom';
const Register = () => {

    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
    });



    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value,
        });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        axios.post('/login', formData).then(resp => {
            console.log(resp)
        })
        console.log('Registration data:', formData);
    };

    return (
        <Container maxWidth="sm">
            <Paper elevation={3} style={{ padding: '20px', marginTop: '100px' }}>
                <Typography variant="h4" component="h1" align="center">
                    Register
                </Typography>
                <form onSubmit={handleSubmit}>
                    <TextField
                        fullWidth
                        variant="outlined"
                        margin="normal"
                        name="username"
                        label="Username"
                        value={formData.username}
                        onChange={handleInputChange}
                        InputProps={{ startAdornment: <Person /> }}
                    />
                    <TextField
                        fullWidth
                        variant="outlined"
                        margin="normal"
                        name="email"
                        label="Email"
                        type="email"
                        value={formData.email}
                        onChange={handleInputChange}
                        InputProps={{ startAdornment: <Email /> }}
                    />
                    <TextField
                        fullWidth
                        variant="outlined"
                        margin="normal"
                        name="password"
                        label="Password"
                        type="password"
                        value={formData.password}
                        onChange={handleInputChange}
                        InputProps={{ startAdornment: <Lock /> }}
                    />
                    {/* <Button type="submit" variant="contained" color="primary" fullWidth>
                        Register
                    </Button> */}
                </form>
                <Button type="submit" variant="contained" color="primary" fullWidth>
                    Register
                </Button>
                <Link to={"/dashboard"}>
                    Dashboard
                </Link>
            </Paper>
        </Container>
    );
};

export default Register;
