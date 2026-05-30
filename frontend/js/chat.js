/**
 * Message list rendering and streaming send handler.
 */

let activeThreadId = null;
let isStreaming = false;
let abortController = null;

const messagesEl = () => document.getElementById("messages");
const statusEl = () => document.getElementById("status-line");
const titleEl = () => document.getElementById("chat-title");
const inputEl = () => document.getElementById("question-input");
const sendBtn = () => document.getElementById("send-btn");

export function getActiveThreadId() {
  return activeThreadId;
}

export function setActiveThreadId(threadId) {
  activeThreadId = threadId;
}

export function clearMessages() {
  messagesEl().innerHTML = "";
  showWelcome();
}

export function showWelcome() {
  const el = messagesEl();
  if (el.children.length === 0) {
    const div = document.createElement("div");
    div.className = "welcome";
    div.textContent = "Ask a question about company policies, profile, or products.";
    el.appendChild(div);
  }
}

export function removeWelcome() {
  const welcome = messagesEl().querySelector(".welcome");
  if (welcome) welcome.remove();
}

export function renderHistory(messages) {
  const el = messagesEl();
  el.innerHTML = "";
  if (!messages.length) {
    showWelcome();
    return;
  }
  for (const msg of messages) {
    appendMessage(msg.role, msg.content, false);
  }
  scrollToBottom();
}

function appendMessage(role, content, streaming = false) {
  removeWelcome();
  const div = document.createElement("div");
  div.className = `message ${role}${streaming ? " streaming" : ""}`;
  div.textContent = content;
  messagesEl().appendChild(div);
  scrollToBottom();
  return div;
}

function scrollToBottom() {
  const el = messagesEl();
  el.scrollTop = el.scrollHeight;
}

export function setStatus(text) {
  statusEl().textContent = text || "";
}

export function setChatTitle(title) {
  titleEl().textContent = title;
}

export function setInputEnabled(enabled) {
  inputEl().disabled = !enabled;
  sendBtn().disabled = !enabled;
}

function showError(message) {
  let banner = document.querySelector(".error-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.className = "error-banner";
    document.querySelector(".chat-main").insertBefore(
      banner,
      document.getElementById("chat-form")
    );
  }
  banner.textContent = message;
  banner.hidden = false;
  setTimeout(() => {
    banner.hidden = true;
  }, 8000);
}

const STATUS_LABELS = {
  deciding: "Thinking…",
  retrieving: "Searching documents…",
  grading: "Reviewing sources…",
  generating: "Generating answer…",
  verifying: "Verifying answer…",
  revising: "Improving answer…",
  checking: "Checking usefulness…",
  rewriting: "Refining search…",
  finishing: "Finishing…",
};

/**
 * @param {{ question: string, streamAsk: Function, onThreadId: (id: string) => void, onDone: () => void, setThreadTitle: (id: string, title: string) => void }} deps
 */
export async function sendMessage(deps) {
  if (isStreaming) return;

  const question = inputEl().value.trim();
  if (!question) return;

  isStreaming = true;
  setInputEnabled(false);
  inputEl().value = "";

  appendMessage("user", question);
  const assistantEl = appendMessage("assistant", "", true);

  abortController = new AbortController();

  let streamThreadId = activeThreadId;
  let streamSucceeded = false;

  try {
    await deps.streamAsk({
      question,
      threadId: activeThreadId,
      signal: abortController.signal,
      onEvent(payload) {
        if (payload.type === "done") {
          streamSucceeded = true;
        }
        switch (payload.type) {
          case "thread_id":
            streamThreadId = payload.thread_id;
            activeThreadId = streamThreadId;
            deps.onThreadId(streamThreadId);
            deps.setThreadTitle(streamThreadId, question);
            break;
          case "status":
            setStatus(STATUS_LABELS[payload.label] || payload.label || "");
            break;
          case "token":
            assistantEl.textContent += payload.content || "";
            scrollToBottom();
            break;
          case "clear":
            assistantEl.textContent = "";
            break;
          case "done":
            assistantEl.classList.remove("streaming");
            if (payload.answer) {
              assistantEl.textContent = payload.answer;
            }
            setStatus("");
            break;
          case "error":
            if (!streamSucceeded && !assistantEl.textContent.trim()) {
              throw new Error(payload.detail || "Stream failed");
            }
            break;
          default:
            break;
        }
      },
    });

    assistantEl.classList.remove("streaming");
    setStatus("");
    deps.onDone();
  } catch (err) {
    assistantEl.classList.remove("streaming");
    if (streamSucceeded || assistantEl.textContent.trim()) {
      // Ignore connection-close noise after a completed answer.
      setStatus("");
      deps.onDone();
    } else if (err.name === "AbortError") {
      assistantEl.textContent += "\n[Cancelled]";
    } else {
      const msg = err.message || String(err);
      if (msg.toLowerCase() !== "network error" || !assistantEl.textContent.trim()) {
        showError(msg);
      }
      if (!assistantEl.textContent) {
        assistantEl.textContent = "Sorry, something went wrong.";
      }
    }
    setStatus("");
  } finally {
    isStreaming = false;
    setInputEnabled(true);
    abortController = null;
    inputEl().focus();
  }
}

export function startNewChat() {
  if (abortController) abortController.abort();
  activeThreadId = null;
  clearMessages();
  setChatTitle("New chat");
  setStatus("");
  inputEl().focus();
}
