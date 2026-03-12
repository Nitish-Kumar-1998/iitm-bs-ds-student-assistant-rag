"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Source {
  doc: string;
  section: string;
  url: string;
  type: string;
  text?: string;
}

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  status?: string;
  loading?: boolean;
  error?: boolean;
  sources?: Source[];
}

const SUGGESTIONS = [
  "What are the eligibility criteria to join?",
  "What is the fee structure for Jan 2026?",
  "How is grading calculated?",
  "What courses are in the Foundation level?",
  "Can I exit with a Diploma?",
  "What happens if I fail a qualifier exam?",
];

function MarkdownContent({ content }: { content: string }) {
  // FIX: strip LLM-generated 📎 Source block — we render sources separately below
  const clean = content.replace(/\n*📎[\s\S]*$/m, "").trim();
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{clean}</ReactMarkdown>
    </div>
  );
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [dark, setDark] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const nextId = useRef(1);

  // FIX: keep a ref in sync with messages so sendQuestion never uses stale closure
  const messagesRef = useRef<Message[]>([]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [question]);

  // FIX: use messagesRef.current so history is always up to date, not stale
  const buildHistory = () =>
    messagesRef.current
      .filter(m => !m.loading && !m.error && m.content)
      .slice(-10)
      .map(m => ({ role: m.role, content: m.content }));

  const sendQuestion = useCallback(async (q: string) => {
    q = q.trim();
    if (!q || isLoading) return;
    const userId = nextId.current++;
    const botId = nextId.current++;
    setMessages(prev => [
      ...prev,
      { id: userId, role: "user", content: q },
      { id: botId, role: "assistant", content: "", loading: true, status: "Thinking..." },
    ]);
    setQuestion("");
    setIsLoading(true);
    try {
      // FIX: call buildHistory() here — uses ref so always current
      const history = buildHistory();
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, history }),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const ev = JSON.parse(line.slice(6));
            if (ev.type === "status") setMessages(p => p.map(m => m.id === botId ? { ...m, status: ev.text } : m));
            if (ev.type === "token") setMessages(p => p.map(m => m.id === botId ? { ...m, content: m.content + ev.text, loading: false, status: undefined } : m));
            if (ev.type === "error") setMessages(p => p.map(m => m.id === botId ? { ...m, content: "⚠️ " + ev.text, loading: false, status: undefined, error: true } : m));
            // FIX 7: capture sources from SSE and attach to message
            if (ev.type === "sources") setMessages(p => p.map(m => m.id === botId ? { ...m, sources: ev.sources } : m));
          } catch {}
        }
      }
    } catch {
      setMessages(p => p.map(m => m.id === botId ? { ...m, content: "⚠️ Could not reach the backend.", loading: false, status: undefined, error: true } : m));
    } finally {
      setIsLoading(false);
      textareaRef.current?.focus();
    }
  }, [isLoading]);

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(question); }
  };

  const copy = (text: string) => navigator.clipboard.writeText(text);

  // FIX: regenerate removes both the last bot message AND the last user message, then resends
  const regenerate = useCallback(() => {
    const msgs = messagesRef.current;
    const lastUser = [...msgs].reverse().find(m => m.role === "user");
    if (!lastUser) return;
    // Remove last assistant + last user message pair
    setMessages(prev => {
      const lastBotIdx = [...prev].map((m, i) => m.role === "assistant" ? i : -1).filter(i => i >= 0).pop();
      if (lastBotIdx === undefined) return prev;
      return prev.slice(0, lastBotIdx - 1 >= 0 ? lastBotIdx : 0);
    });
    setTimeout(() => sendQuestion(lastUser.content), 50);
  }, [sendQuestion]);

  const t = dark ? {
    bg: "#212121", sb: "#171717", sbBorder: "rgba(255,255,255,.07)",
    text: "#ececec", textMuted: "rgba(255,255,255,.45)", textFaint: "rgba(255,255,255,.2)",
    inputBg: "#2f2f2f", inputBorder: "rgba(255,255,255,.1)", inputBorderFocus: "rgba(255,255,255,.25)",
    btnHover: "#2a2a2a", divider: "rgba(255,255,255,.08)",
    msgUser: "rgba(255,255,255,.75)", codeBg: "#2a2a2a",
    tableBg: "rgba(255,255,255,.06)", tableText: "rgba(255,255,255,.65)",
    chipBg: "rgba(255,255,255,.05)", chipBorder: "rgba(255,255,255,.1)",
    topbarBorder: "rgba(255,255,255,.06)",
  } : {
    bg: "#ffffff", sb: "#f9f9f9", sbBorder: "rgba(0,0,0,.08)",
    text: "#1a1a1a", textMuted: "rgba(0,0,0,.45)", textFaint: "rgba(0,0,0,.25)",
    inputBg: "#f4f4f4", inputBorder: "rgba(0,0,0,.12)", inputBorderFocus: "rgba(0,0,0,.3)",
    btnHover: "#f0f0f0", divider: "rgba(0,0,0,.08)",
    msgUser: "rgba(0,0,0,.7)", codeBg: "#f0f0f0",
    tableBg: "rgba(0,0,0,.04)", tableText: "rgba(0,0,0,.6)",
    chipBg: "rgba(0,0,0,.04)", chipBorder: "rgba(0,0,0,.1)",
    topbarBorder: "rgba(0,0,0,.07)",
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
        html,body{height:100%;font-family:'Inter',sans-serif;-webkit-font-smoothing:antialiased;overflow:hidden}
        .app{display:flex;height:100vh;width:100vw;background:${t.bg};color:${t.text};transition:background .2s,color .2s}

        /* SIDEBAR */
        .sb{width:${sidebarOpen ? "260px" : "0px"};flex-shrink:0;background:${t.sb};border-right:1px solid ${t.sbBorder};display:flex;flex-direction:column;overflow:hidden;transition:width .25s cubic-bezier(.4,0,.2,1)}
        .sb-inner{width:260px;display:flex;flex-direction:column;height:100%;padding:10px 8px}
        .nb{display:flex;align-items:center;gap:10px;width:100%;padding:10px 12px;border-radius:8px;border:none;background:transparent;color:${t.text};font-size:.875rem;font-family:'Inter',sans-serif;cursor:pointer;transition:background .15s;margin-bottom:4px}
        .nb:hover{background:${t.btnHover}}
        .sdiv{height:1px;background:${t.divider};margin:8px 4px 12px}
        .sbot{margin-top:auto;padding:12px 12px 4px;border-top:1px solid ${t.divider}}
        .sbadge{display:flex;align-items:center;gap:8px;padding:8px 0}
        .sicon{width:28px;height:28px;border-radius:6px;background:#c8102e;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#fff;flex-shrink:0}
        .sname{font-size:.78rem;font-weight:500;color:${t.textMuted}}
        .ssub{font-size:.63rem;color:${t.textFaint}}
        .sfooter{font-size:.65rem;color:${t.textFaint};line-height:1.6;margin-top:4px}

        /* MAIN */
        .main{flex:1;display:flex;flex-direction:column;min-width:0;background:${t.bg};transition:background .2s}

        /* TOPBAR */
        .tb{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid ${t.topbarBorder};flex-shrink:0;gap:12px}
        .tb-left{display:flex;align-items:center;gap:10px}
        .toggle-btn{width:32px;height:32px;border-radius:7px;border:none;background:transparent;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s;flex-shrink:0}
        .toggle-btn:hover{background:${t.btnHover}}
        .toggle-btn svg{opacity:.6}
        .tm{font-size:.9rem;font-weight:500;color:${t.text};display:flex;align-items:center;gap:8px}
        .tdot{width:7px;height:7px;border-radius:50%;background:#22c55e;animation:pulse 2s infinite}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        .tb-right{display:flex;align-items:center;gap:8px}
        .tbtn{padding:5px 12px;background:transparent;border:1px solid ${t.inputBorder};border-radius:6px;color:${t.textMuted};font-size:.75rem;font-family:'Inter',sans-serif;cursor:pointer;transition:all .15s}
        .tbtn:hover{background:${t.btnHover};color:${t.text}}

        /* THEME TOGGLE */
        .theme-btn{width:32px;height:32px;border-radius:7px;border:none;background:transparent;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s}
        .theme-btn:hover{background:${t.btnHover}}

        /* MESSAGES */
        .msgs{flex:1;overflow-y:auto;padding:24px 0;scrollbar-width:thin;scrollbar-color:${t.divider} transparent}
        .msgs::-webkit-scrollbar{width:6px}
        .msgs::-webkit-scrollbar-thumb{background:${t.divider};border-radius:3px}

        /* EMPTY */
        .empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:24px;padding:40px;animation:up .4s ease forwards}
        @keyframes up{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        .eh{font-size:1.75rem;font-weight:600;color:${t.text};text-align:center;line-height:1.3}
        .eh span{color:#c8102e}
        .ep{font-size:.875rem;color:${t.textMuted};text-align:center;max-width:420px;line-height:1.75}
        .chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:640px}
        .chip{padding:8px 16px;background:${t.chipBg};border:1px solid ${t.chipBorder};border-radius:20px;color:${t.textMuted};font-size:.8rem;cursor:pointer;transition:all .2s;font-family:'Inter',sans-serif}
        .chip:hover{background:${t.btnHover};color:${t.text};border-color:#c8102e;transform:translateY(-1px)}

        /* MSG — FIX: max-width increased to 900px for better readability */
        .msg{max-width:900px;margin:0 auto;padding:16px 24px;animation:up .2s ease forwards;opacity:0}
        .mh{display:flex;align-items:center;gap:10px;margin-bottom:10px}
        .av{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:600;flex-shrink:0}
        .av.bot{background:#c8102e;color:#fff}
        .av.usr{background:${dark ? "#4a4a4a" : "#e0e0e0"};color:${dark ? "rgba(255,255,255,.8)" : "rgba(0,0,0,.6)"}}
        .mn{font-size:.82rem;font-weight:600;color:${t.textMuted}}
        .mb{font-size:.9375rem;line-height:1.75;color:${t.text};padding-left:38px}
        .mb.ub{color:${t.msgUser}}
        .mb.eb{color:#f87171}

        /* DOTS */
        .dots{display:flex;gap:4px;align-items:center;padding:4px 0}
        .dots span{width:6px;height:6px;border-radius:50%;background:${t.textFaint};animation:d 1.2s infinite}
        .dots span:nth-child(2){animation-delay:.15s}
        .dots span:nth-child(3){animation-delay:.3s}
        @keyframes d{0%,80%,100%{opacity:.2;transform:scale(.85)}40%{opacity:1;transform:scale(1)}}
        .stxt{font-size:.8rem;color:${t.textFaint};font-style:italic;display:flex;align-items:center;gap:8px}

        /* ACTIONS */
        .ma{display:flex;gap:6px;margin-top:10px;padding-left:38px;opacity:0;transition:opacity .15s}
        .msg:hover .ma{opacity:1}
        .ab{padding:4px 10px;background:none;border:1px solid ${t.inputBorder};border-radius:6px;color:${t.textMuted};font-size:.72rem;cursor:pointer;font-family:'Inter',sans-serif;transition:all .15s}
        .ab:hover{border-color:${t.inputBorderFocus};color:${t.text}}

        /* MARKDOWN */
        .md{width:100%}
        .md p{margin-bottom:.75em}
        .md p:last-child{margin-bottom:0}
        .md h1,.md h2,.md h3{font-weight:600;color:${t.text};margin:1.1em 0 .4em}
        .md h2{font-size:1.1rem}
        .md h3{font-size:.975rem}
        .md ul,.md ol{padding-left:1.5em;margin-bottom:.75em}
        .md li{margin-bottom:.3em}
        .md strong{font-weight:600;color:${t.text}}
        .md em{font-style:italic;color:${t.textMuted}}
        .md code{font-family:'JetBrains Mono',monospace;font-size:.82em;background:${t.codeBg};color:${dark?"#e0a0a0":"#c8102e"};padding:1px 6px;border-radius:4px}
        .md pre{background:${t.codeBg};border-radius:8px;padding:14px 16px;margin:12px 0;overflow-x:auto;border:1px solid ${t.inputBorder}}
        .md pre code{background:none;color:${dark?"#d4d4d4":"#333"};padding:0;font-size:.85rem}
        .md table{width:100%;border-collapse:collapse;margin:12px 0;font-size:.875rem}
        .md th{background:${t.tableBg};color:${t.text};padding:9px 13px;text-align:left;border:1px solid ${t.inputBorder};font-weight:500;font-size:.8rem}
        .md td{padding:9px 13px;border:1px solid ${t.inputBorder};color:${t.tableText};vertical-align:top}
        .md tr:nth-child(even) td{background:${t.tableBg}}
        .md a{color:#c8102e;text-decoration:underline;text-underline-offset:3px}
        .md blockquote{border-left:3px solid ${t.divider};padding:6px 14px;margin:10px 0;color:${t.textMuted};font-style:italic}
        .md hr{border:none;border-top:1px solid ${t.divider};margin:14px 0}

        /* FIX 7: source links */
        .src-list{display:flex;flex-direction:column;gap:4px;padding-left:38px;margin-top:10px;margin-bottom:4px}
        .src-link{font-size:.75rem;color:#c8102e;text-decoration:none;opacity:.8;transition:opacity .15s;word-break:break-all;line-height:1.5}
        .src-link:hover{opacity:1;text-decoration:underline}

        /* FIX 8: improved status text */
        .stxt{font-size:.8rem;color:${t.textFaint};font-style:italic;display:flex;align-items:center;gap:8px;padding:2px 0}
        .stxt::before{content:"";width:6px;height:6px;border-radius:50%;background:#c8102e;animation:pulse 1s infinite;flex-shrink:0}

        /* INPUT — FIX: max-width matched to message width */
        .ia{padding:12px 24px 20px;flex-shrink:0;max-width:900px;width:100%;margin:0 auto}
        .ib{position:relative;background:${t.inputBg};border-radius:16px;border:1px solid ${t.inputBorder};transition:border-color .2s;padding:14px 52px 14px 18px}
        .ib:focus-within{border-color:${t.inputBorderFocus}}
        .ib textarea{width:100%;background:transparent;border:none;outline:none;color:${t.text};font-size:.9375rem;font-family:'Inter',sans-serif;resize:none;line-height:1.6;max-height:200px;overflow-y:auto;scrollbar-width:thin;display:block}
        .ib textarea::placeholder{color:${t.textFaint}}
        .sb2{position:absolute;bottom:10px;right:10px;width:32px;height:32px;border-radius:8px;border:none;background:#c8102e;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:background .15s,transform .1s,opacity .15s}
        .sb2:hover:not(:disabled){background:#a50d25;transform:scale(1.05)}
        .sb2:disabled{opacity:.35;cursor:not-allowed}
        .sb2 svg{width:13px;height:13px;fill:#fff}
        .ih{text-align:center;font-size:.68rem;color:${t.textFaint};margin-top:8px}
      `}</style>

      <div className="app">
        {/* SIDEBAR */}
        <aside className="sb">
          <div className="sb-inner">
            <button className="nb" onClick={() => setMessages([])}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              New chat
            </button>
            <div className="sdiv"/>

            <div style={{ padding: "0 12px", fontSize: ".68rem", fontWeight: 500, color: t.textFaint, letterSpacing: ".8px", textTransform: "uppercase", marginBottom: "8px" }}>Today</div>
            {messages.length > 0 && (
              <div style={{ padding: "8px 12px", fontSize: ".82rem", color: t.textMuted, borderRadius: "8px", background: dark ? "rgba(255,255,255,.06)" : "rgba(0,0,0,.05)", cursor: "default", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {messages.find(m => m.role === "user")?.content.slice(0, 30) || "New conversation"}...
              </div>
            )}

            <div className="sbot">
              <div className="sbadge">
                <div className="sicon">IITM</div>
                <div>
                  {/* FIX: label now matches header — "BS Programme Assistant" */}
                  <div className="sname">BS Programme Assistant</div>
                  <div className="ssub">IIT Madras · Jan 2026</div>
                </div>
              </div>
              <div className="sfooter">RAG · Qdrant · Groq</div>
            </div>
          </div>
        </aside>

        {/* MAIN */}
        <main className="main">
          {/* TOPBAR */}
          <div className="tb">
            <div className="tb-left">
              <button className="toggle-btn" onClick={() => setSidebarOpen(o => !o)} title="Toggle sidebar">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={t.text} strokeWidth="2">
                  <rect x="3" y="3" width="18" height="18" rx="2"/>
                  <line x1="9" y1="3" x2="9" y2="21"/>
                </svg>
              </button>
              <div className="tm">
                <span className="tdot"/>
                IITM BS Programme Assistant
              </div>
            </div>
            <div className="tb-right">
              <button className="theme-btn" onClick={() => setDark(d => !d)} title={dark ? "Light mode" : "Dark mode"}>
                {dark ? (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={t.textMuted} strokeWidth="2">
                    <circle cx="12" cy="12" r="5"/>
                    <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                    <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={t.textMuted} strokeWidth="2">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                  </svg>
                )}
              </button>
              {messages.length > 0 && (
                <button className="tbtn" onClick={() => setMessages([])}>Clear chat</button>
              )}
            </div>
          </div>

          {/* MESSAGES */}
          <div className="msgs">
            {messages.length === 0 ? (
              <div className="empty">
                {/* FIX: grammar fixed */}
                <h1 className="eh">What can I help you with<br/>about <span>IITM BS Programme?</span></h1>
                <p className="ep">Ask anything about courses, fees, eligibility, grading, or deadlines — answered from official documents.</p>
                <div className="chips">
                  {SUGGESTIONS.map(s => (
                    <button key={s} className="chip" onClick={() => sendQuestion(s)}>{s}</button>
                  ))}
                </div>
              </div>
            ) : messages.map((msg, idx) => (
              <div key={msg.id} className="msg">
                <div className="mh">
                  <div className={`av ${msg.role === "assistant" ? "bot" : "usr"}`}>
                    {msg.role === "assistant" ? "AI" : "U"}
                  </div>
                  <span className="mn">{msg.role === "assistant" ? "IITM Assistant" : "You"}</span>
                </div>
                <div className={`mb ${msg.role === "user" ? "ub" : ""} ${msg.error ? "eb" : ""}`}>
                  {msg.loading && !msg.content ? (
                    msg.status
                      ? <div className="stxt">{msg.status}<div className="dots"><span/><span/><span/></div></div>
                      : <div className="dots"><span/><span/><span/></div>
                  ) : msg.role === "assistant"
                    ? <MarkdownContent content={msg.content}/>
                    : msg.content}
                </div>
                {msg.role === "assistant" && !msg.loading && msg.content && (
                  <>
                    {/* FIX 7: clickable source URLs rendered below answer */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="src-list">
                        {msg.sources.filter(s => s.url).map((s, si) => (
                          <a key={si} href={s.url} target="_blank" rel="noopener noreferrer" className="src-link">
                            📎 {s.section ? `${s.section} — ` : ""}{s.doc || s.text || "Source"}
                          </a>
                        ))}
                      </div>
                    )}
                    <div className="ma">
                      <button className="ab" onClick={() => copy(msg.content)}>Copy</button>
                      {idx === messages.length - 1 && (
                        <button className="ab" onClick={regenerate}>Regenerate</button>
                      )}
                    </div>
                  </>
                )}
              </div>
            ))}
            <div ref={bottomRef}/>
          </div>

          {/* INPUT */}
          <div className="ia">
            <div className="ib">
              <textarea
                ref={textareaRef}
                rows={1}
                placeholder="Message IITM BS Assistant..."
                value={question}
                onChange={e => setQuestion(e.target.value)}
                onKeyDown={handleKey}
                disabled={isLoading}
                maxLength={500}
              />
              <button className="sb2" onClick={() => sendQuestion(question)} disabled={!question.trim() || isLoading}>
                <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
              </button>
            </div>
            <div className="ih">Enter to send · Shift+Enter for new line</div>
          </div>
        </main>
      </div>
    </>
  );
}