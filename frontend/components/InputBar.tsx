"use client";

import { useRef, useEffect, useState } from "react";

interface InputBarProps {
  dark: boolean;
  isLoading: boolean;
  onSend: (q: string) => void;
}

export default function InputBar({ dark, isLoading, onSend }: InputBarProps) {
  const [question, setQuestion] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const MAX = 500;

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [question]);

  const handleSend = () => {
    if (!question.trim() || isLoading) return;
    onSend(question);
    setQuestion("");
  };

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const remaining = MAX - question.length;
  const nearLimit = remaining < 50;

  return (
    <div className={`px-4 pb-safe pb-5 pt-3 flex-shrink-0 ${dark ? "bg-[#212121]" : "bg-white"}`}>
      <div className="max-w-3xl mx-auto">
        <div className={`relative rounded-2xl border transition-all ${
          dark
            ? "bg-[#2f2f2f] border-white/10 focus-within:border-white/25"
            : "bg-gray-100 border-black/10 focus-within:border-black/25"
        }`}>
          <textarea
            ref={textareaRef}
            rows={1}
            placeholder="Message IITM BS DS Assistant..."
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKey}
            disabled={isLoading}
            maxLength={MAX}
            className={`w-full bg-transparent border-none outline-none resize-none px-4 py-3.5 pr-14 text-sm leading-relaxed max-h-48 overflow-y-auto ${
              dark ? "text-white/90 placeholder:text-white/20" : "text-gray-800 placeholder:text-gray-400"
            } disabled:opacity-50 text-base sm:text-sm`}
          />
          <button
            onClick={handleSend}
            disabled={!question.trim() || isLoading}
            className="absolute bottom-2.5 right-2.5 w-8 h-8 rounded-lg bg-[#c8102e] flex items-center justify-center transition-all active:scale-90 hover:bg-[#a50d25] disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="white">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>

        {/* Footer */}
        <div className="flex justify-between items-center mt-1.5 px-1">
          <span className={`text-xs ${dark ? "text-white/20" : "text-gray-400"}`}>
            Enter to send · Shift+Enter for new line
          </span>
          {nearLimit && (
            <span className={`text-xs ${remaining < 20 ? "text-red-400" : dark ? "text-white/30" : "text-gray-400"}`}>
              {remaining}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}