import React from 'react';
import { FileText, Brain, Sparkles } from 'lucide-react';

const AboutPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-black pt-20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent mb-6">
              About Policy Assistant
            </h1>
            <p className="text-xl text-slate-400 leading-relaxed">
              Revolutionizing insurance policy management through artificial intelligence
            </p>
          </div>

          <div className="space-y-12">
            <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-3xl p-8 border border-slate-700/30">
              <h2 className="text-3xl font-bold text-white mb-6">Our Mission</h2>
              <p className="text-lg text-slate-300 leading-relaxed mb-6">
                Policy Assistant is designed to transform the complex world of insurance policy management. 
                We leverage cutting-edge artificial intelligence to make policy analysis accessible, accurate, and instantaneous.
              </p>
              <p className="text-lg text-slate-300 leading-relaxed">
                Our platform bridges the gap between complex policy documents and clear, actionable decisions, 
                empowering users to understand their coverage with unprecedented clarity.
              </p>
            </div>

            <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-3xl p-8 border border-slate-700/30">
              <h2 className="text-3xl font-bold text-white mb-6">How It Works</h2>
              <div className="grid md:grid-cols-3 gap-8">
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Brain className="w-8 h-8 text-cyan-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-3">AI Analysis</h3>
                  <p className="text-slate-400">Our advanced AI models analyze your documents and understand complex policy language.</p>
                </div>
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-pink-500/20 to-pink-600/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Sparkles className="w-8 h-8 text-pink-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-3">Get Answers</h3>
                  <p className="text-slate-400">Ask questions in natural language and receive instant, accurate policy decisions.</p>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-3xl p-8 border border-slate-700/30">
              <h2 className="text-3xl font-bold text-white mb-6">Key Technologies</h2>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-xl font-semibold text-purple-300 mb-3">Natural Language Processing</h3>
                  <p className="text-slate-400 mb-4">Advanced NLP models understand and interpret complex policy language, making it accessible to everyone.</p>
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-cyan-300 mb-3">Machine Learning</h3>
                  <p className="text-slate-400 mb-4">Trained on thousands of policy documents to ensure accurate and reliable decision-making.</p>
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-pink-300 mb-3">Document Processing</h3>
                  <p className="text-slate-400 mb-4">Intelligent document parsing and indexing for fast, comprehensive policy analysis.</p>
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-emerald-300 mb-3">Security First</h3>
                  <p className="text-slate-400 mb-4">Enterprise-grade security ensures your sensitive policy information remains protected.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;