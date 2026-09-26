"use client";

import { useRef, useState } from "react";

type Msg = { role: "user" | "assistant"; content: string };

export default function Home() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  function appendToLast(text: string) {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      next[next.length - 1] = { ...last, content: last.content + text };
      return next;
    });
  }

  async function sendMessage() {
    const query = input.trim();
    if (!query || loading) return;

    setMessages((m) => [...m, { role: "user", content: query }, { role: "assistant", content: "" }]);
    setInput("");
    setLoading(true);

    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
        signal: controller.signal,
      });

      if (res.status === 429) {
        const data = await res.json().catch(() => ({}));
        appendToLast(data.error ?? "Rate limited. Try again shortly.");
        return;
      }

      if (!res.ok || !res.body) {
        appendToLast("Something went wrong reaching ArXAgent.");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        appendToLast(decoder.decode(value, { stream: true }));
      }
    } catch {
      appendToLast("Connection interrupted.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main
      style={{
        maxWidth: 720,
        margin: "0 auto",
        padding: "2rem 1rem",
        fontFamily: "system-ui, -apple-system, sans-serif",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <h1 style={{ fontSize: "1.4rem", marginBottom: "1.25rem", color: "#111827" }}>
        ArXAgent
      </h1>

      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", flex: 1, marginBottom: "1.5rem" }}>
        {messages.length === 0 && (
          <p style={{ color: "#6b7280", fontSize: "0.9rem" }}>
            Ask about a paper, topic, or recent research to get started.
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              alignSelf: m.role === "user" ? "flex-end" : "flex-start",
              background: m.role === "user" ? "#111827" : "#f3f4f6",
              color: m.role === "user" ? "#fff" : "#111827",
              padding: "0.6rem 0.9rem",
              borderRadius: 10,
              maxWidth: "85%",
              whiteSpace: "pre-wrap",
              fontSize: "0.92rem",
              lineHeight: 1.45,
            }}
          >
            {m.content || (loading && m.role === "assistant" && i === messages.length - 1 ? "…" : "")}
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: "0.5rem" }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask ArXAgent about a paper or topic…"
          style={{
            flex: 1,
            padding: "0.65rem 0.9rem",
            borderRadius: 8,
            border: "1px solid #d1d5db",
            fontSize: "0.92rem",
          }}
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          style={{
            padding: "0.65rem 1.2rem",
            borderRadius: 8,
            background: "#111827",
            color: "#fff",
            border: "none",
            fontSize: "0.92rem",
            cursor: loading ? "default" : "pointer",
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? "…" : "Send"}
        </button>
      </div>
    </main>
  );
}
