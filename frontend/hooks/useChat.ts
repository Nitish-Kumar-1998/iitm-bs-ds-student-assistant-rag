"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Message } from "@/types";

const STORAGE_KEY = "iitm_bs_ds_chat";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [dark, setDark] = useState(false);
  const messagesRef = useRef<Message[]>([]);
  const nextId = useRef(1);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        setMessages(parsed);
        nextId.current = parsed.length
          ? Math.max(...parsed.map((m: Message) => m.id)) + 1
          : 1;
      }
    } catch {}
  }, []);

  // Save to localStorage on every message change
  useEffect(() => {
    messagesRef.current = messages;
    try {
      // Only save non-loading messages
      const toSave = messages.filter(m => !m.loading);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
    } catch {}
  }, [messages]);

  const buildHistory = () =>
    messagesRef.current
      .filter(m => !m.loading && !m.error && m.content)
      .slice(-10)
      .map(m => ({ role: m.role, content: m.content }));

  const clearChat = useCallback(() => {
    setMessages([]);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

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
    setIsLoading(true);

    try {
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
            if (ev.type === "status")
              setMessages(p => p.map(m => m.id === botId ? { ...m, status: ev.text } : m));
            if (ev.type === "token")
              setMessages(p => p.map(m => m.id === botId ? { ...m, content: m.content + ev.text, loading: false, status: undefined } : m));
            if (ev.type === "error")
              setMessages(p => p.map(m => m.id === botId ? { ...m, content: "⚠️ " + ev.text, loading: false, status: undefined, error: true } : m));
            if (ev.type === "sources")
              setMessages(p => p.map(m => m.id === botId ? { ...m, sources: ev.sources } : m));
          } catch {}
        }
      }
    } catch {
      setMessages(p => p.map(m => m.id === botId ? { ...m, content: "⚠️ Could not reach the backend.", loading: false, status: undefined, error: true } : m));
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const regenerate = useCallback((messageId: number) => {
    const msgs = messagesRef.current;
    const msgIndex = msgs.findIndex(m => m.id === messageId);
    if (msgIndex === -1) return;
    // Find the user message before this bot message
    const userMsg = [...msgs].slice(0, msgIndex).reverse().find(m => m.role === "user");
    if (!userMsg) return;
    // Remove from this bot message onwards
    setMessages(prev => prev.slice(0, msgIndex));
    setTimeout(() => sendQuestion(userMsg.content), 50);
  }, [sendQuestion]);

  return {
    messages,
    isLoading,
    dark,
    setDark,
    sendQuestion,
    regenerate,
    clearChat,
  };
}