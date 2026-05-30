import { listChats, fetchChatHistory } from "./api.js";
import {
  getThreadTitle,
  sortThreadIds,
  touchThread,
} from "./storage.js";
import {
  getActiveThreadId,
  setActiveThreadId,
  renderHistory,
  setChatTitle,
  clearMessages,
} from "./chat.js";

const listEl = () => document.getElementById("thread-list");

export async function refreshThreadList() {
  let threadIds = [];
  try {
    threadIds = await listChats();
  } catch {
    threadIds = [];
  }

  const sorted = sortThreadIds(threadIds);
  const active = getActiveThreadId();
  const nav = listEl();
  nav.innerHTML = "";

  if (!sorted.length) {
    const empty = document.createElement("p");
    empty.className = "thread-empty";
    empty.style.cssText = "padding:8px 12px;color:var(--text-muted);font-size:13px;margin:0;";
    empty.textContent = "No chats yet";
    nav.appendChild(empty);
    return;
  }

  for (const id of sorted) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thread-item" + (id === active ? " active" : "");
    btn.textContent = getThreadTitle(id);
    btn.dataset.threadId = id;
    btn.addEventListener("click", () => loadThread(id));
    nav.appendChild(btn);
  }
}

export async function loadThread(threadId) {
  setActiveThreadId(threadId);
  touchThread(threadId);
  setChatTitle(getThreadTitle(threadId));
  clearMessages();

  try {
    const data = await fetchChatHistory(threadId);
    renderHistory(data.messages || []);
  } catch (err) {
    renderHistory([]);
    console.error(err);
  }

  await refreshThreadList();
}

export function highlightActiveThread(threadId) {
  listEl().querySelectorAll(".thread-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.threadId === threadId);
  });
}
