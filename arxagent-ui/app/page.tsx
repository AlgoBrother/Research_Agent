"use client";

import { useRef, useState } from "react";

type Paper = {
  title: string | null;
  pdf_url: string | null;
  authors: string[];
  published: string | null;
  summary: string | null;
  arxiv_id: string | null;
};

type Msg = {
  role: "user" | "assistant";
  content: string;
  status?: string;
  papers?: Paper[];
};

// A random-enough per-tab session id, kept for the life of the tab, so the
// backend can keep a stateful ResearchAgent per session (conversation
// history, session memory) instead of one shared instance.
function getSessionId(): string {
  if (typeof window === "undefined") return "server";
  const key = "arxagent_session_id";
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(key, id);
  }
  return id;
}

export default function Home() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  function updateLast(patch: Partial<Msg>) {
    setMessages((prev) => {
      const next = [...prev];
      next[next.length - 1] = { ...next[next.length - 1], ...patch };
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
        body: JSON.stringify({ query, session_id: getSessionId() }),
        signal: controller.signal,
      });

      if (res.status === 429) {
        const data = await res.json().catch(() => ({}));
        updateLast({ content: data.error ?? "Rate limited. Try again shortly.", status: undefined });
        return;
      }

      if (!res.ok || !res.body) {
        updateLast({ content: "Something went wrong reaching ArXAgent.", status: undefined });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line; each starts with "data: ".
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? ""; // last (possibly incomplete) frame stays buffered

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          const jsonStr = line.slice("data:".length).trim();
          if (!jsonStr) continue;

          let event: any;
          try {
            event = JSON.parse(jsonStr);
          } catch {
            continue;
          }

          if (event.type === "step") {
            updateLast({ status: event.message });
          } else if (event.type === "answer") {
            updateLast({ content: event.answer, status: undefined, papers: event.papers ?? [] });
          } else if (event.type === "error") {
            updateLast({ content: `Error: ${event.message}`, status: undefined });
          }
        }
      }
    } catch {
      updateLast({ content: "Connection interrupted.", status: undefined });
    } finally {
      setLoading(false);
    }
  }

  return (
    <main
      style={{
        maxWidth: 760,
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

      <div style={{ display: "flex", flexDirection: "column", gap: "0.9rem", flex: 1, marginBottom: "1.5rem" }}>
        {messages.length === 0 && (
          <p style={{ color: "#6b7280", fontSize: "0.9rem" }}>
            Ask about a paper, topic, or recent research to get started.
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ alignSelf: m.role === "user" ? "flex-end" : "flex-start", maxWidth: "88%" }}>
            <div
              style={{
                background: m.role === "user" ? "#111827" : "#f3f4f6",
                color: m.role === "user" ? "#fff" : "#111827",
                padding: "0.6rem 0.9rem",
                borderRadius: 10,
                whiteSpace: "pre-wrap",
                fontSize: "0.92rem",
                lineHeight: 1.45,
              }}
            >
              {m.content || (m.status ? `⏳ ${m.status}` : loading && i === messages.length - 1 ? "…" : "")}
            </div>

            {m.papers && m.papers.length > 0 && (
              <details style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "#374151" }}>
                <summary style={{ cursor: "pointer", color: "#6b7280" }}>
                  {m.papers.length} source paper{m.papers.length > 1 ? "s" : ""}
                </summary>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem", marginTop: "0.5rem" }}>
                  {m.papers.map((p, j) => (
                    <div key={j} style={{ borderLeft: "2px solid #e5e7eb", paddingLeft: "0.6rem" }}>
                      {p.pdf_url ? (
                        <a href={p.pdf_url} target="_blank" rel="noreferrer" style={{ fontWeight: 600, color: "#111827" }}>
                          {p.title}
                        </a>
                      ) : (
                        <span style={{ fontWeight: 600 }}>{p.title}</span>
                      )}
                      <div style={{ color: "#6b7280", fontSize: "0.8rem" }}>
                        {p.authors?.slice(0, 2).join(", ")}
                        {p.authors && p.authors.length > 2 ? " et al." : ""}
                        {p.published ? ` · ${p.published.slice(0, 10)}` : ""}
                      </div>
                      {p.summary && <div style={{ fontSize: "0.82rem", marginTop: "0.2rem" }}>{p.summary.slice(0, 200)}…</div>}
                    </div>
                  ))}
                </div>
              </details>
            )}
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