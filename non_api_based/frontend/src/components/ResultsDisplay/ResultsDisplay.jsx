import React from 'react';
import { CheckCircle } from 'lucide-react';
import { renderJsonValue } from '../../utils/jsonRenderer';

const ResultsDisplay = ({ response }) => {
  if (!response) return null;

  return (
    <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-2xl p-4 sm:p-6 lg:p-8 border border-slate-700/30 animate-in slide-in-from-bottom-4 duration-500">
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 sm:gap-4 mb-4 sm:mb-6">
        <div className="p-2 sm:p-3 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-lg sm:rounded-xl">
          <CheckCircle className="w-5 h-5 sm:w-6 sm:h-6 lg:w-7 lg:h-7 text-emerald-400" />
        </div>
        <div className="flex-1">
          <h2 className="text-lg sm:text-xl lg:text-2xl font-semibold text-slate-200">Analysis Results</h2>
          <p className="text-sm sm:text-base text-slate-500">AI-powered decision and justification</p>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 mb-6 sm:mb-8">
        <div className="bg-gradient-to-br from-emerald-900/30 to-emerald-800/30 p-3 sm:p-4 rounded-lg sm:rounded-xl border border-emerald-500/30">
          <div className="text-emerald-400 font-semibold text-xs sm:text-sm">Decision</div>
          <div className="text-emerald-300 text-base sm:text-lg font-bold">{response.decision}</div>
        </div>
        <div className="bg-gradient-to-br from-cyan-900/30 to-cyan-800/30 p-3 sm:p-4 rounded-lg sm:rounded-xl border border-cyan-500/30">
          <div className="text-cyan-400 font-semibold text-xs sm:text-sm">Coverage</div>
          <div className="text-cyan-300 text-base sm:text-lg font-bold">{response.coverage_amount}</div>
        </div>
        <div className="bg-gradient-to-br from-purple-900/30 to-purple-800/30 p-3 sm:p-4 rounded-lg sm:rounded-xl border border-purple-500/30">
          <div className="text-purple-400 font-semibold text-xs sm:text-sm">Confidence</div>
          <div className="text-purple-300 text-base sm:text-lg font-bold">
            {response.confidence_score ? (response.confidence_score * 100).toFixed(0) : 'N/A'}%
          </div>
        </div>
      </div>

      {/* Detailed JSON Response */}
      <div className="bg-gradient-to-br from-black/50 to-slate-900/50 rounded-xl sm:rounded-2xl p-3 sm:p-4 lg:p-6 border border-slate-700/50 shadow-inner overflow-hidden">
        <div className="flex items-center gap-2 mb-3 sm:mb-4 text-slate-400">
          <div className="w-2 h-2 sm:w-3 sm:h-3 bg-red-500 rounded-full"></div>
          <div className="w-2 h-2 sm:w-3 sm:h-3 bg-yellow-500 rounded-full"></div>
          <div className="w-2 h-2 sm:w-3 sm:h-3 bg-green-500 rounded-full"></div>
          <span className="ml-1 sm:ml-2 text-xs sm:text-sm font-mono">response.json</span>
        </div>
        <div className="overflow-x-auto">
          <pre className="text-xs sm:text-sm font-mono text-slate-300 whitespace-pre-wrap">
            {renderJsonValue(response)}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;