import React, { useState } from 'react';
import { AppBar, Toolbar, Typography, Container, Paper } from '@mui/material';
import { makeStyles } from '@mui/styles';
import { TextField, Button } from '@mui/material';
import axios from 'axios';

const useStyles = makeStyles((theme) => ({
    root: {
        display: 'flex',
    }, 
}));

function Dashboard() {
    const [searchText, setSearchText] = useState('');
    const classes = useStyles();

    const handleSearch = () => {
        axios.get(`/predict?text=${searchText}`).then(res => {
            console.log(res)
        })
    }

    return (
        <div className={classes.root}>
            <AppBar position="fixed" className={classes.appBar}>
                <Toolbar>
                    <Typography variant="h6">Dashboard</Typography>
                </Toolbar>
            </AppBar>
            <main className={classes.content} style={{ marginTop: '100px' }}>
                <Container>
                    <Paper elevation={3} style={{ padding: 16 }}>
                        <div style={{ marginBottom: 16 , justifyContent: 'center' }}>
                            <Typography>
                                Search By Your Symptoms
                            </Typography>
                            <TextField
                                style={{ marginTop: '12px' }}
                                label="Search"
                                fullWidth
                                size='small'
                                value={searchText}
                                onChange={(e) => setSearchText(e.target.value)}
                            />
                            <Button
                                variant="contained"
                                color="primary"
                                style={{ marginTop: '12px' }}
                                onClick={handleSearch}
                            >
                                Search
                            </Button>
                        </div>
                    </Paper>
                </Container>
            </main>
        </div>
    );
}

export default Dashboard;
