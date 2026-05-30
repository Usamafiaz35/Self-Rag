import { API_BASE_URL } from "./config.js";

async function parseJsonResponse(res) {
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      const body = JSON.parse(text);
      detail = body.detail ?? text;
    } catch {
      /* use raw text */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  return parseJsonResponse(res);
}

export async function listChats() {
  const res = await fetch(`${API_BASE_URL}/chats`);
  const data = await parseJsonResponse(res);
  return data.thread_ids || [];
}

export async function fetchChatHistory(threadId) {
  const res = await fetch(`${API_BASE_URL}/chats/${encodeURIComponent(threadId)}`);
  return parseJsonResponse(res);
}

/**
 * POST /ask/stream and invoke onEvent for each SSE data payload.
 * @param {{ question: string, threadId?: string|null, onEvent: (payload: object) => void, signal?: AbortSignal }} opts
 */
function dispatchSseEvent(payload, onEvent) {
  onEvent(payload);
}

function parseSseBuffer(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const rest = parts.pop() || "";
  for (const part of parts) {
    const line = part.split("\n").find((l) => l.startsWith("data: "));
    if (!line) continue;
    try {
      dispatchSseEvent(JSON.parse(line.slice(6)), onEvent);
    } catch {
      /* skip malformed chunk */
    }
  }
  return rest;
}

export async function streamAsk({ question, threadId, onEvent, signal }) {
  const res = await fetch(`${API_BASE_URL}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      thread_id: threadId || null,
    }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let receivedDone = false;

  const handlePayload = (payload) => {
    if (payload.type === "done") {
      receivedDone = true;
    }
    onEvent(payload);
  };

  try {
    while (true) {
      let readResult;
      try {
        readResult = await reader.read();
      } catch (readErr) {
        // Chrome/Firefox may throw "network error" when the server closes SSE after done.
        if (receivedDone) {
          break;
        }
        throw readErr;
      }

      const { done, value } = readResult;
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      buffer = parseSseBuffer(buffer, handlePayload);
    }

    if (buffer.trim()) {
      parseSseBuffer(`${buffer}\n\n`, handlePayload);
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* already released */
    }
  }
}
