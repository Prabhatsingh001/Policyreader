import React from 'react';
import { Upload, FileText, Zap, CheckCircle } from 'lucide-react';

const DocumentUpload = ({ onFileUpload, documentsUploaded }) => {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && onFileUpload) {
      onFileUpload(file);
    }
  };

  return (
    <div className={`group bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-2xl p-4 sm:p-6 lg:p-8 border transition-all duration-500 ${
      documentsUploaded 
        ? 'border-emerald-500/50 hover:border-emerald-400/70' 
        : 'border-slate-700/30 hover:border-purple-500/30'
    }`}>
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 sm:gap-4 mb-4 sm:mb-6">
        <div className={`p-2 sm:p-3 rounded-lg sm:rounded-xl ${
          documentsUploaded 
            ? 'bg-gradient-to-br from-emerald-500/20 to-green-600/20' 
            : 'bg-gradient-to-br from-cyan-500/20 to-blue-600/20'
        }`}>
          {documentsUploaded ? (
            <CheckCircle className="w-5 h-5 sm:w-6 sm:h-6 lg:w-7 lg:h-7 text-emerald-400" />
          ) : (
            <Upload className="w-5 h-5 sm:w-6 sm:h-6 lg:w-7 lg:h-7 text-cyan-400" />
          )}
        </div>
        <div className="flex-1">
          <h2 className="text-lg sm:text-xl lg:text-2xl font-semibold text-slate-200">
            {documentsUploaded ? 'Documents Ready' : 'Document Upload'}
          </h2>
          <p className="text-sm sm:text-base text-slate-500">
            {documentsUploaded 
              ? 'Documents processed and ready for analysis' 
              : 'Process and analyze policy documents'
            }
          </p>
        </div>
      </div>
      
      {documentsUploaded && (
        <div className="mb-4 sm:mb-6 p-3 sm:p-4 bg-gradient-to-r from-emerald-900/30 to-green-800/30 border border-emerald-500/30 rounded-xl sm:rounded-2xl">
          <div className="flex items-center gap-3">
            <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />
            <div>
              <p className="text-emerald-300 font-semibold text-sm sm:text-base">Documents Successfully Uploaded</p>
              <p className="text-emerald-200/80 text-xs sm:text-sm">You can now submit queries for AI analysis</p>
            </div>
          </div>
        </div>
      )}
      
      <div className="relative group/upload">
        <input
          type="file"
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          accept=".pdf,.doc,.docx,.txt"
        />
        <div className={`text-white px-4 sm:px-6 lg:px-8 py-3 sm:py-4 rounded-xl sm:rounded-2xl cursor-pointer transition-all duration-300 flex flex-col sm:flex-row items-center gap-2 sm:gap-4 justify-center shadow-lg group-hover/upload:scale-105 ${
          documentsUploaded
            ? 'bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-400 hover:to-green-500 hover:shadow-emerald-500/25 hover:shadow-2xl'
            : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 hover:shadow-cyan-500/25 hover:shadow-2xl'
        }`}>
          <FileText className="w-5 h-5 sm:w-6 sm:h-6" />
          <span className="font-semibold text-base sm:text-lg">
            {documentsUploaded ? 'Upload More Documents' : 'Select Documents'}
          </span>
          <Zap className="w-4 h-4 sm:w-5 sm:h-5 opacity-70" />
        </div>
      </div>
      
      <p className="text-xs sm:text-sm text-slate-500 mt-3 sm:mt-4 text-center opacity-70 px-2">
        Supports PDF, Word, and text documents • Backend integration required for processing
      </p>
    </div>
  );
};

export default DocumentUpload;

// import React from 'react';
// import { Upload, FileText, Zap } from 'lucide-react';

// const DocumentUpload = ({ onFileUpload }) => {
//   const handleFileChange = (event) => {
//     const file = event.target.files[0];
//     if (file && onFileUpload) {
//       onFileUpload(file);
//     }
//   };

//   return (
//     <div className="group bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-2xl p-4 sm:p-6 lg:p-8 border border-slate-700/30 hover:border-purple-500/30 transition-all duration-500">
//       <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 sm:gap-4 mb-4 sm:mb-6">
//         <div className="p-2 sm:p-3 bg-gradient-to-br from-cyan-500/20 to-blue-600/20 rounded-lg sm:rounded-xl">
//           <Upload className="w-5 h-5 sm:w-6 sm:h-6 lg:w-7 lg:h-7 text-cyan-400" />
//         </div>
//         <div className="flex-1">
//           <h2 className="text-lg sm:text-xl lg:text-2xl font-semibold text-slate-200">Document Upload</h2>
//           <p className="text-sm sm:text-base text-slate-500">Process and analyze policy documents</p>
//         </div>
//       </div>
      
//       <div className="relative group/upload">
//         <input
//           type="file"
//           onChange={handleFileChange}
//           className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
//           accept=".pdf,.doc,.docx,.txt"
//         />
//         <div className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white px-4 sm:px-6 lg:px-8 py-3 sm:py-4 rounded-xl sm:rounded-2xl cursor-pointer transition-all duration-300 flex flex-col sm:flex-row items-center gap-2 sm:gap-4 justify-center shadow-lg hover:shadow-cyan-500/25 hover:shadow-2xl group-hover/upload:scale-105">
//           <FileText className="w-5 h-5 sm:w-6 sm:h-6" />
//           <span className="font-semibold text-base sm:text-lg">Select Documents</span>
//           <Zap className="w-4 h-4 sm:w-5 sm:h-5 opacity-70" />
//         </div>
//       </div>
      
//       <p className="text-xs sm:text-sm text-slate-500 mt-3 sm:mt-4 text-center opacity-70 px-2">
//         Supports PDF, Word, and text documents • Backend integration required for processing
//       </p>
//     </div>
//   );
// };

// export default DocumentUpload;