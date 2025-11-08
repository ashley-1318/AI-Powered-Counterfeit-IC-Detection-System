import React, { useContext } from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link, useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';

function Navigation() {
  const { user, logout } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <AppBar position="fixed">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          CircuitCheck
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button 
            color="inherit" 
            component={Link} 
            to="/"
          >
            Dashboard
          </Button>
          <Button 
            color="inherit" 
            component={Link} 
            to="/analysis"
          >
            Analysis
          </Button>
          <Button 
            color="inherit" 
            component={Link} 
            to="/results"
          >
            Results
          </Button>
          {user ? (
            <>
              <Button 
                color="inherit"
                onClick={handleLogout}
              >
                Logout ({user.username})
              </Button>
            </>
          ) : (
            <Button 
              color="inherit" 
              component={Link} 
              to="/login"
            >
              Login
            </Button>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Navigation;