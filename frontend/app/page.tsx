"use client";

import TopBar from "@/components/TopBar";
import ChatWindow from "@/components/ChatWindow";
import InputBar from "@/components/InputBar";
import { useChat } from "@/hooks/useChat";

export default function Home() {
  const { messages, isLoading, dark, setDark, sendQuestion, regenerate, clearChat } = useChat();

  return (
    <div
      style={{ height: "100dvh" }}
      className={`flex flex-col w-screen overflow-hidden ${dark ? "bg-[#212121]" : "bg-white"}`}
    >
      <TopBar
        dark={dark}
        setDark={setDark}
        hasMessages={messages.length > 0}
        clearChat={clearChat}
      />
      <ChatWindow
        messages={messages}
        dark={dark}
        isLoading={isLoading}
        onRegenerate={regenerate}
        onSuggestion={sendQuestion}
      />
      <InputBar
        dark={dark}
        isLoading={isLoading}
        onSend={sendQuestion}
      />
    </div>
  );
}