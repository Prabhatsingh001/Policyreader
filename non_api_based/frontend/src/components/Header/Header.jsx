import React from 'react';
import { Brain, Sparkles } from 'lucide-react';

const Header = () => {
  return (
    <div className="text-center mb-8 sm:mb-12 lg:mb-16">
      <div className="inline-flex items-center justify-center gap-2 sm:gap-4 mb-4 sm:mb-6 p-3 sm:p-4 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 backdrop-blur-sm rounded-xl sm:rounded-2xl border border-purple-500/20">
        <Brain className="w-8 h-8 sm:w-10 sm:h-10 lg:w-12 lg:h-12 text-purple-400 animate-pulse" />
        <div className="w-px h-8 sm:h-10 lg:h-12 bg-gradient-to-b from-transparent via-slate-400 to-transparent"></div>
        <Sparkles className="w-6 h-6 sm:w-8 sm:h-8 lg:w-10 lg:h-10 text-cyan-400" />
      </div>
      <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent mb-3 sm:mb-4">
        Policy Assistant
      </h1>
      <p className="text-base sm:text-lg lg:text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed px-4">
        Advanced AI-powered document analysis and intelligent policy decision-making system
      </p>
    </div>
  );
};

export default Header;