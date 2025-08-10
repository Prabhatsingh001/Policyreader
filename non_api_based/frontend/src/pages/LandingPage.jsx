import React from 'react';
import { Brain, Sparkles, ArrowRight, Shield, Zap, Award } from 'lucide-react';
import BackgroundEffects from '../components/common/BackgroundEffects';

const LandingPage = ({ setCurrentPage }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-black relative overflow-hidden">
      <BackgroundEffects />

      <div className="relative z-10 pt-20">
        {/* Hero Section */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-flex items-center gap-4 mb-8 p-4 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 backdrop-blur-sm rounded-2xl border border-purple-500/20">
              <Brain className="w-12 h-12 text-purple-400 animate-pulse" />
              <div className="w-px h-12 bg-gradient-to-b from-transparent via-slate-400 to-transparent"></div>
              <Sparkles className="w-10 h-10 text-cyan-400" />
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent mb-6">
              Policy Assistant
            </h1>
            
            <p className="text-xl md:text-2xl text-slate-400 mb-8 leading-relaxed">
              Revolutionary AI-powered document analysis and intelligent policy decision-making system
            </p>
            
            <p className="text-lg text-slate-500 mb-12 max-w-2xl mx-auto">
              Transform your insurance policy management with cutting-edge artificial intelligence. 
              Upload documents, ask questions in natural language, and get instant, accurate policy decisions.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
              <button
                onClick={() => setCurrentPage('app')}
                className="group bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-400 hover:to-pink-500 text-white px-8 py-4 rounded-2xl font-semibold text-lg transition-all duration-300 flex items-center gap-3 shadow-lg hover:shadow-purple-500/30 hover:shadow-2xl transform hover:scale-105"
              >
                Try Policy Assistant
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
              
              <button
                onClick={() => setCurrentPage('about')}
                className="px-8 py-4 rounded-2xl font-semibold text-lg border border-slate-600 text-slate-300 hover:border-purple-500/50 hover:text-white transition-all duration-300"
              >
                Learn More
              </button>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-2xl p-8 border border-slate-700/30 hover:border-purple-500/30 transition-all duration-500">
              <Shield className="w-12 h-12 text-purple-400 mb-6" />
              <h3 className="text-xl font-semibold text-white mb-4">Secure Analysis</h3>
              <p className="text-slate-400">Advanced encryption and secure processing ensure your sensitive policy documents remain protected.</p>
            </div>
            
            <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-2xl p-8 border border-slate-700/30 hover:border-cyan-500/30 transition-all duration-500">
              <Zap className="w-12 h-12 text-cyan-400 mb-6" />
              <h3 className="text-xl font-semibold text-white mb-4">Lightning Fast</h3>
              <p className="text-slate-400">Get instant policy decisions powered by state-of-the-art AI models and optimized processing.</p>
            </div>
            
            <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-2xl p-8 border border-slate-700/30 hover:border-pink-500/30 transition-all duration-500">
              <Award className="w-12 h-12 text-pink-400 mb-6" />
              <h3 className="text-xl font-semibold text-white mb-4">Highly Accurate</h3>
              <p className="text-slate-400">Machine learning algorithms trained on thousands of policies ensure precise decision-making.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;