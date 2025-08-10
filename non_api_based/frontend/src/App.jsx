import React, { useState } from 'react';
import Navigation from './components/common/Navigation';
import LandingPage from './pages/LandingPage';
import PolicyAssistantApp from './pages/PolicyAssistantApp';
import AboutPage from './pages/AboutPage';
import TeamPage from './pages/TeamPage';
import './styles/animations.css';

const App = () => {
  const [currentPage, setCurrentPage] = useState('home');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'home':
        return <LandingPage setCurrentPage={setCurrentPage} />;
      case 'app':
        return <PolicyAssistantApp />;
      case 'about':
        return <AboutPage />;
      case 'team':
        return <TeamPage />;
      default:
        return <LandingPage setCurrentPage={setCurrentPage} />;
    }
  };

  return (
    <div className="font-['Inter']">
      <Navigation 
        currentPage={currentPage} 
        setCurrentPage={setCurrentPage}
        isMobileMenuOpen={isMobileMenuOpen}
        setIsMobileMenuOpen={setIsMobileMenuOpen}
      />
      {renderCurrentPage()}
    </div>
  );
};

export default App;