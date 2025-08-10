import React, { useState, useEffect } from 'react';
import { Brain, Sparkles, FileText, Upload, Send, Paperclip, X } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';

const PolicyAssistantApp = () => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [pendingFiles, setPendingFiles] = useState([]);
  const [sessionId, setSessionId] = useState('');
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const initiateSession = async () => {
      const existingSession = localStorage.getItem("sessionId");

      if (existingSession) {
        try {
          const validateRes = await fetch(`http://localhost:3000/api/chat/validate/${existingSession}`);
          const validateData = await validateRes.json();

          if (validateData.valid) {
            setSessionId(existingSession);
            console.log("Reusing existing session:", existingSession);
            return;
          } else {
            console.log("Old session expired or invalid, creating new...");
          }
        } catch (err) {
          console.error("Error validating session:", err);
        }
      }

      try {
        const res = await fetch("http://localhost:3000/api/chat/initiate");
        const data = await res.json();
        if (data.sessionId) {
          setSessionId(data.sessionId);
          localStorage.setItem("sessionId", data.sessionId);
          console.log("New session created:", data.sessionId);
        } else {
          throw new Error("No sessionId in response");
        }
      } catch (err) {
        console.log("Session initiation failed:", err);
        setError("Failed to start session. Please try again later.");
      }
    };

    initiateSession();
  }, []);

  const handleFileUpload = async (file) => {
    console.log(`Uploading file: "${file.name}"`);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("sessionId", localStorage.getItem("sessionId"));

    try {
      const response = await fetch("http://localhost:3000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();
      console.log("Upload response:", data);

      setPendingFiles((prev) => [...prev, file.name]);
      setError("");
    } catch (err) {
      console.error("Upload error:", err);
      setError("File upload failed. Please try again.");
    }
  };

  const removeFile = (index) => {
    const newFiles = pendingFiles.filter((_, i) => i !== index);
    setPendingFiles(newFiles);
  };

  const handleQuerySubmit = async () => {
    if (messages.length === 0 && pendingFiles.length === 0) {
      setError('Please upload at least one document for the first message.');
      return;
    }

    if (!query.trim()) {
      setError('Please enter a query before submitting.');
      return;
    }

    setIsLoading(true);
    setError('');

    const currentFiles = [...pendingFiles];
    setMessages(prev => [...prev, { sender: 'user', text: query, files: currentFiles }]);

    try {
      const formData = new FormData();
      formData.append('chat', query);
      formData.append('sessionId', sessionId);

      const fileInputs = document.querySelectorAll('input[type="file"]');
      fileInputs.forEach(input => {
        if (input.files[0]) {
          formData.append('documents', input.files[0]);
        }
      });

      const res = await fetch('http://localhost:3000/chat/send', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      setMessages(prev => [...prev, {
        sender: 'ai',
        text: data.ai_response || "No response from AI.",
        files: []
      }]);

      setPendingFiles([]);
      fileInputs.forEach(input => input.value = '');

    } catch (err) {
      console.error('Query submission error:', err);
      setMessages(prev => [...prev, { sender: 'ai', text: 'Error processing your request.', files: [] }]);
    } finally {
      setQuery('');
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuerySubmit();
    }
  };

  return (
    <div className="h-screen overflow-hidden bg-gradient-to-br from-slate-900 via-gray-900 to-black font-['Inter'] relative">
      <style jsx>{`
        ::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        ::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
          background: linear-gradient(180deg, #a855f7 0%, #06b6d4 100%);
          border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(180deg, #9333ea 0%, #0891b2 100%);
        }
      `}</style>
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-20 -right-20 sm:-top-40 sm:-right-40 w-40 h-40 sm:w-80 sm:h-80 bg-purple-500/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-20 -left-20 sm:-bottom-40 sm:-left-40 w-40 h-40 sm:w-80 sm:h-80 bg-cyan-500/10 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-48 h-48 sm:w-96 sm:h-96 bg-pink-500/5 rounded-full blur-3xl"></div>
      </div>

      <div className="container mx-auto px-3 sm:px-4 md:px-6 lg:px-8 pt-13 sm:pt-17 pb-4 sm:pb-6 max-w-4xl relative z-10 h-full flex flex-col">
        {messages.length === 0 && (
          <div className="text-center mb-4 sm:mb-6 mt-35">
            <div className="inline-flex items-center justify-center gap-2 sm:gap-3 mb-3 sm:mb-4 p-2 sm:p-3 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 backdrop-blur-sm rounded-xl border border-purple-500/20">
              <Brain className="w-6 h-6 sm:w-8 sm:h-8 text-purple-400 animate-pulse" />
              <div className="w-px h-6 sm:h-8 bg-gradient-to-b from-transparent via-slate-400 to-transparent"></div>
              <Sparkles className="w-5 h-5 sm:w-7 sm:h-7 text-cyan-400" />
            </div>
            <h1 className="text-xl sm:text-3xl md:text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent mb-2">
              Policy Assistant
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 max-w-xl mx-auto leading-relaxed px-4">
              Advanced AI-powered document analysis and intelligent policy decision-making
            </p>
          </div>
        )}

        <div
          className="flex-1 overflow-y-auto px-2 sm:px-4 space-y-4 scrollbar-thin mt-0 mb-1"
          style={{ maxHeight: messages.length === 0 ? 'calc(100vh - 200px - 120px - 24px)' : 'calc(100vh - 120px - 24px)' }}
        >
          {messages.map((msg, index) => (
            <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'} items-start`}>
              <div className="flex flex-col items-end w-fit max-w-[80%] sm:max-w-[70%]">
                <div
                  className={`block rounded-2xl px-3 py-2 text-sm leading-relaxed shadow-md w-fit min-w-0 overflow-wrap-break-word box-border ${
                    msg.sender === 'user'
                      ? 'bg-gradient-to-br from-purple-600 to-pink-600 text-white rounded-br-none'
                      : 'bg-slate-800 border border-slate-700 text-slate-200 rounded-bl-none'
                  }`}
                >
                  {msg.text}
                </div>
                {msg.sender === 'user' && msg.files && msg.files.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-2 w-full justify-end">
                    {msg.files.map((fileName, fileIndex) => (
                      <div
                        key={fileIndex}
                        className="flex items-center gap-2 bg-emerald-800/30 border border-emerald-500/40 px-3 py-1 rounded-lg"
                      >
                        <FileText className="w-3 h-3 text-emerald-400" />
                        <span className="text-emerald-200 text-xs">
                          {fileName.length > 15 ? `${fileName.substring(0, 15)}...` : fileName}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-auto">
          {error && (
            <div className="mb-3 p-3 bg-red-900/40 border border-red-500/40 rounded-lg">
              <p className="text-red-300 text-sm overflow-wrap-break-word">{error}</p>
            </div>
          )}

          {pendingFiles.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              {pendingFiles.map((fileName, index) => (
                <div key={index} className="flex items-center gap-2 bg-emerald-800/30 border border-emerald-500/40 px-3 py-1 rounded-lg">
                  <FileText className="w-3 h-3 text-emerald-400" />
                  <span className="text-emerald-200 text-xs">
                    {fileName.length > 15 ? `${fileName.substring(0, 15)}...` : fileName}
                  </span>
                  <button
                    onClick={() => removeFile(index)}
                    className="text-emerald-400 hover:text-emerald-300 ml-1"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="bg-gradient-to-br from-slate-800/60 to-slate-900/60 backdrop-blur-xl rounded-2xl shadow-xl border border-slate-700/40 p-3">
            <div className="flex items-center gap-3">
              <div className="relative flex-shrink-0">
                <input
                  type="file"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileUpload(e.target.files[0]);
                    }
                  }}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  accept=".pdf,.doc,.docx,.txt"
                />
                <button className="w-10 h-10 bg-slate-700/80 hover:bg-slate-600/80 border border-slate-600/50 hover:border-slate-500/50 rounded-xl transition-all duration-200 flex items-center justify-center">
                  <Paperclip className="w-5 h-5 text-slate-300" />
                </button>
              </div>

              <div className="flex-1 relative">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about policy coverage, claims, or specific scenarios..."
                  className="w-full min-h-[60px] max-h-32 p-3 bg-slate-800/50 border border-slate-600/30 rounded-xl resize-none transition-all duration-300 text-slate-200 text-sm placeholder-slate-500 focus:ring-2 focus:ring-purple-500/50 focus:border-transparent pr-12 scrollbar-thin outline-none"
                  maxLength={500}
                  rows={3}
                />
                <div className="absolute bottom-2 right-2 text-xs text-slate-500">
                  {query.length}/500
                </div>
              </div>

              <button
                onClick={handleQuerySubmit}
                disabled={isLoading || !query.trim() || (messages.length === 0 && pendingFiles.length === 0)}
                className={`w-10 h-10 rounded-xl transition-all duration-300 flex items-center justify-center flex-shrink-0 ${
                  query.trim() && (messages.length > 0 || pendingFiles.length > 0)
                    ? 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white shadow-lg hover:shadow-purple-500/25'
                    : 'bg-slate-700/80 text-slate-400 cursor-not-allowed'
                }`}
              >
                {isLoading ? (
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>

            <div className="mt-2 text-xs text-slate-500 text-center">
              {messages.length === 0 && pendingFiles.length === 0 ? (
                <span className="text-amber-400">Upload at least one document to start</span>
              ) : (
                <span>Ready for analysis • {pendingFiles.length} document{pendingFiles.length !== 1 ? 's' : ''} selected</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PolicyAssistantApp;