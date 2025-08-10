import React from 'react';
import { Search, Brain, Sparkles, Loader2, AlertCircle } from 'lucide-react';
import ErrorMessage from '../common/ErrorMessage';
import LoadingSpinner from '../common/LoadingSpinner';

const QueryInterface = ({ 
  query, 
  setQuery, 
  onSubmit, 
  isLoading, 
  error, 
  documentsUploaded 
}) => {
  const handleSubmit = () => {
    if (!documentsUploaded) {
      return;
    }
    if (onSubmit) {
      onSubmit();
    }
  };

  return (
    <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-2xl p-4 sm:p-6 lg:p-8 border border-slate-700/30">
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 sm:gap-4 mb-4 sm:mb-6">
        <div className="p-2 sm:p-3 bg-gradient-to-br from-purple-500/20 to-pink-600/20 rounded-lg sm:rounded-xl">
          <Search className="w-5 h-5 sm:w-6 sm:h-6 lg:w-7 lg:h-7 text-purple-400" />
        </div>
        <div className="flex-1">
          <h2 className="text-lg sm:text-xl lg:text-2xl font-semibold text-slate-200">Intelligent Query</h2>
          <p className="text-sm sm:text-base text-slate-500">
            {documentsUploaded 
              ? 'Natural language policy analysis' 
              : 'Upload documents first to enable analysis'
            }
          </p>
        </div>
      </div>

      <div className="space-y-4 sm:space-y-6">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={!documentsUploaded}
            placeholder={
              documentsUploaded 
                ? "Describe your scenario in natural language... e.g., '46-year-old male needs knee surgery in Pune with a 3-month-old insurance policy'"
                : "Please upload documents first to enable query analysis"
            }
            className={`w-full h-32 sm:h-36 lg:h-40 p-4 sm:p-5 lg:p-6 bg-gradient-to-br from-slate-800/50 to-slate-900/50 border rounded-xl sm:rounded-2xl resize-none transition-all duration-300 text-slate-200 text-sm sm:text-base lg:text-lg leading-relaxed ${
              documentsUploaded 
                ? 'border-slate-600/50 focus:ring-2 focus:ring-purple-500/50 focus:border-transparent placeholder-slate-500' 
                : 'border-red-500/50 placeholder-red-400/70 cursor-not-allowed opacity-60'
            }`}
            maxLength={500}
          />
          <div className={`absolute bottom-3 sm:bottom-4 right-3 sm:right-4 text-xs sm:text-sm ${
            documentsUploaded ? 'text-slate-500' : 'text-red-400/70'
          }`}>
            {documentsUploaded ? `${query.length}/500` : 'Disabled'}
          </div>
        </div>

        <button
          onClick={handleSubmit}
          disabled={isLoading || !documentsUploaded}
          className={`w-full px-4 sm:px-6 lg:px-8 py-4 sm:py-5 rounded-xl sm:rounded-2xl font-semibold text-base sm:text-lg transition-all duration-300 flex flex-col sm:flex-row items-center gap-2 sm:gap-4 justify-center shadow-lg transform hover:scale-105 disabled:hover:scale-100 disabled:cursor-not-allowed ${
            documentsUploaded 
              ? 'bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-400 hover:to-pink-500 disabled:from-slate-600 disabled:to-slate-700 text-white hover:shadow-purple-500/30 hover:shadow-2xl'
              : 'bg-gradient-to-r from-slate-600 to-slate-700 text-slate-400 cursor-not-allowed opacity-60'
          }`}
        >
          {isLoading ? (
            <LoadingSpinner />
          ) : (
            <>
              <Brain className="w-5 h-5 sm:w-6 sm:h-6" />
              <span>{documentsUploaded ? 'Analyze with AI' : 'Upload Documents First'}</span>
              {documentsUploaded && <Sparkles className="w-4 h-4 sm:w-5 sm:h-5 opacity-70" />}
            </>
          )}
        </button>

        {!documentsUploaded && (
          <div className="p-3 sm:p-4 bg-gradient-to-r from-amber-900/30 to-orange-800/30 border border-amber-500/50 rounded-xl sm:rounded-2xl flex items-start gap-3 backdrop-blur-sm">
            <AlertCircle className="w-5 h-5 sm:w-6 sm:h-6 text-amber-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-amber-300 text-sm sm:text-base font-semibold mb-1">No Documents Uploaded</p>
              <p className="text-amber-200/80 text-xs sm:text-sm">Please upload policy documents first to enable AI analysis and querying. The system needs documents to provide accurate policy decisions.</p>
            </div>
          </div>
        )}

        {error && documentsUploaded && <ErrorMessage message={error} />}
      </div>
    </div>
  );
};

export default QueryInterface;
// import React from 'react';
// import { Search, Brain, Sparkles, Loader2 } from 'lucide-react';
// import ErrorMessage from '../common/ErrorMessage';
// import LoadingSpinner from '../common/LoadingSpinner';

// const QueryInterface = ({ query, setQuery, onSubmit, isLoading, error }) => {
//   const handleSubmit = () => {
//     if (onSubmit) {
//       onSubmit();
//     }
//   };

//   return (
//     <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-2xl p-4 sm:p-6 lg:p-8 border border-slate-700/30">
//       <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 sm:gap-4 mb-4 sm:mb-6">
//         <div className="p-2 sm:p-3 bg-gradient-to-br from-purple-500/20 to-pink-600/20 rounded-lg sm:rounded-xl">
//           <Search className="w-5 h-5 sm:w-6 sm:h-6 lg:w-7 lg:h-7 text-purple-400" />
//         </div>
//         <div className="flex-1">
//           <h2 className="text-lg sm:text-xl lg:text-2xl font-semibold text-slate-200">Intelligent Query</h2>
//           <p className="text-sm sm:text-base text-slate-500">Natural language policy analysis</p>
//         </div>
//       </div>

//       <div className="space-y-4 sm:space-y-6">
//         <div className="relative">
//           <textarea
//             value={query}
//             onChange={(e) => setQuery(e.target.value)}
//             placeholder="Describe your scenario in natural language... e.g., '46-year-old male needs knee surgery in Pune with a 3-month-old insurance policy'"
//             className="w-full h-32 sm:h-36 lg:h-40 p-4 sm:p-5 lg:p-6 bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-slate-600/50 rounded-xl sm:rounded-2xl resize-none focus:ring-2 focus:ring-purple-500/50 focus:border-transparent transition-all duration-300 text-slate-200 placeholder-slate-500 text-sm sm:text-base lg:text-lg leading-relaxed"
//             maxLength={500}
//           />
//           <div className="absolute bottom-3 sm:bottom-4 right-3 sm:right-4 text-slate-500 text-xs sm:text-sm">
//             {query.length}/500
//           </div>
//         </div>

//         <button
//           onClick={handleSubmit}
//           disabled={isLoading}
//           className="w-full bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-400 hover:to-pink-500 disabled:from-slate-600 disabled:to-slate-700 text-white px-4 sm:px-6 lg:px-8 py-4 sm:py-5 rounded-xl sm:rounded-2xl font-semibold text-base sm:text-lg transition-all duration-300 flex flex-col sm:flex-row items-center gap-2 sm:gap-4 justify-center shadow-lg hover:shadow-purple-500/30 hover:shadow-2xl disabled:cursor-not-allowed transform hover:scale-105 disabled:hover:scale-100"
//         >
//           {isLoading ? (
//             <LoadingSpinner />
//           ) : (
//             <>
//               <Brain className="w-5 h-5 sm:w-6 sm:h-6" />
//               <span>Analyze with AI</span>
//               <Sparkles className="w-4 h-4 sm:w-5 sm:h-5 opacity-70" />
//             </>
//           )}
//         </button>

//         {error && <ErrorMessage message={error} />}
//       </div>
//     </div>
//   );
// };

// export default QueryInterface;