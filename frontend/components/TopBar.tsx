"use client";

interface TopBarProps {
  dark: boolean;
  setDark: (d: boolean) => void;
  hasMessages: boolean;
  clearChat: () => void;
}

export default function TopBar({ dark, setDark, hasMessages, clearChat }: TopBarProps) {
  return (
    <div className={`flex items-center justify-between px-4 py-3 border-b flex-shrink-0 ${dark ? "border-white/10 bg-[#212121]" : "border-black/8 bg-white"}`}>
      {/* Left */}
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
        <span className={`text-sm font-medium ${dark ? "text-white/90" : "text-gray-800"}`}>
          IITM BS DS Assistant
        </span>
      </div>

      {/* Right */}
      <div className="flex items-center gap-2">
        {/* Theme toggle */}
        <button
          onClick={() => setDark(!dark)}
          className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all active:scale-95 ${dark ? "hover:bg-white/10 text-white/60" : "hover:bg-black/8 text-gray-500"}`}
          title={dark ? "Light mode" : "Dark mode"}
        >
          {dark ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="5"/>
              <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
              <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
          )}
        </button>

        {/* Clear chat */}
        {hasMessages && (
          <button
            onClick={clearChat}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-all active:scale-95 ${dark ? "border-white/15 text-white/50 hover:bg-white/10 hover:text-white/80" : "border-black/12 text-gray-500 hover:bg-black/6 hover:text-gray-700"}`}
          >
            Clear chat
          </button>
        )}
      </div>
    </div>
  );
}