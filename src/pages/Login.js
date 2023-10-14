import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Container, Paper, Typography, TextField, Button } from '@mui/material';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = (e) => {
    e.preventDefault();
    // Add your login logic here
    console.log('Email:', email);
    console.log('Password:', password);
  };

  return (
    <Container maxWidth="xs">
      <Paper elevation={3} style={{ padding: '20px', marginTop: '100px' }}>
        <Typography
          variant="h6"
          sx={{ textAlign: 'center' }}
          gutterBottom>
          Login
        </Typography>
        <form onSubmit={handleLogin}>
          <TextField
            label="Email"
            variant="outlined"
            fullWidth
            size='small'
            margin="normal"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <TextField
            label="Password"
            variant="outlined"
            type="password"
            fullWidth
            size='small'
            margin="normal"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <Button
            sx={{ marginTop: '12px' }}
            variant="contained"
            color="primary"
            type="submit"
            fullWidth>
            Login
          </Button>
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            marginTop: '12px'
          }}>
            <Link to={'/register'}>
              <Typography>
                Create Account
              </Typography>
            </Link>
          </div>
        </form>
      </Paper>
    </Container>
  );
};

export default Login;
