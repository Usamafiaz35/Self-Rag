import { streamAsk, fetchHealth } from "./api.js";
import { sendMessage, startNewChat, setChatTitle } from "./chat.js";
import { refreshThreadList, highlightActiveThread } from "./sidebar.js";
import { setThreadTitle, getThreadTitle } from "./storage.js";

const form = document.getElementById("chat-form");
const input = document.getElementById("question-input");
const newChatBtn = document.getElementById("new-chat-btn");

function autoResizeTextarea() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
}

input.addEventListener("input", autoResizeTextarea);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

newChatBtn.addEventListener("click", () => {
  startNewChat();
  refreshThreadList();
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage({
    streamAsk,
    onThreadId(threadId) {
      highlightActiveThread(threadId);
      setChatTitle(getThreadTitle(threadId));
    },
    onDone() {
      refreshThreadList();
    },
    setThreadTitle,
  });
});

async function init() {
  try {
    await fetchHealth();
  } catch {
    const banner = document.createElement("div");
    banner.className = "error-banner";
    banner.textContent =
      "Cannot reach the API. Start the backend from project root: python scripts/run_api.py";
    document.querySelector(".chat-main").insertBefore(
      banner,
      document.getElementById("chat-form")
    );
  }

  startNewChat();
  await refreshThreadList();
  input.focus();
}

init();
