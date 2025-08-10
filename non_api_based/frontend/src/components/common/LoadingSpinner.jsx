import React from 'react';

const LoadingSpinner = ({ text = "Analyzing Query..." }) => {
  return (
    <>
      <div className="w-6 h-6 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
      <span>{text}</span>
      <div className="flex gap-1">
        <div className="w-2 h-2 bg-white/60 rounded-full animate-bounce"></div>
        <div className="w-2 h-2 bg-white/60 rounded-full animate-bounce delay-100"></div>
        <div className="w-2 h-2 bg-white/60 rounded-full animate-bounce delay-200"></div>
      </div>
    </>
  );
};

export default LoadingSpinner;