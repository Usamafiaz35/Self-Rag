const META_KEY = "self_rag_chat_meta";

/**
 * @returns {Record<string, { title: string, lastActive: number }>}
 */
function loadMeta() {
  try {
    const raw = localStorage.getItem(META_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveMeta(meta) {
  localStorage.setItem(META_KEY, JSON.stringify(meta));
}

export function getThreadTitle(threadId) {
  const meta = loadMeta();
  return meta[threadId]?.title || shortId(threadId);
}

export function setThreadTitle(threadId, title) {
  const meta = loadMeta();
  const existing = meta[threadId] || { title: "", lastActive: 0 };
  if (!existing.title) {
    existing.title = truncate(title, 40);
  }
  existing.lastActive = Date.now();
  meta[threadId] = existing;
  saveMeta(meta);
}

export function touchThread(threadId) {
  const meta = loadMeta();
  const existing = meta[threadId] || { title: shortId(threadId), lastActive: 0 };
  existing.lastActive = Date.now();
  meta[threadId] = existing;
  saveMeta(meta);
}

export function sortThreadIds(threadIds) {
  const meta = loadMeta();
  return [...threadIds].sort((a, b) => {
    const ta = meta[a]?.lastActive ?? 0;
    const tb = meta[b]?.lastActive ?? 0;
    return tb - ta;
  });
}

function truncate(text, max) {
  const t = text.trim();
  if (t.length <= max) return t;
  return t.slice(0, max - 1) + "…";
}

function shortId(threadId) {
  return `Chat ${threadId.slice(0, 8)}`;
}
