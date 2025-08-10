import React from 'react';

const ErrorMessage = ({ message }) => {
  if (!message) return null;

  return (
    <div className="p-4 bg-gradient-to-r from-red-900/30 to-red-800/30 border border-red-500/50 rounded-2xl flex items-start gap-3 backdrop-blur-sm">
      <div className="w-6 h-6 text-red-400 mt-0.5 flex-shrink-0">⚠</div>
      <p className="text-red-300">{message}</p>
    </div>
  );
};

export default ErrorMessage;