import React from 'react';
import { Navigate } from 'react-router-dom';

// Since the focus is on the personalized ML feed, we just redirect the home route to ForYou.
// Alternatively, Home could be a generic non-personalized trending feed.
const Home = () => {
  return <Navigate to="/foryou" replace />;
};

export default Home;
