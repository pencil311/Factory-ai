import type { ChatRequest, StreamEvent } from "@/lib/types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const TENANT_ID = import.meta.env.VITE_TENANT_ID ?? "demo";

export interface ChatStreamHandle {
  /** Yields one parsed SSE frame at a time; ends cleanly if the server closes
   * the connection without ever sending a `done` event. */
  events: AsyncGenerator<StreamEvent>;
  /** Abort the underlying fetch — the server observes the disconnect via
   * `request.is_disconnected` and stops the orchestration mid-flight. */
  controller: AbortController;
}

/** POST /chat/stream as a live event stream.
 *
 * EventSource cannot send a POST body, so this drives the SSE wire format
 * (`event: <type>` / `data: <json>` frames separated by a blank line, `:`
 * lines are heartbeat comments) by hand over a streamed fetch response.
 */
export function streamChat(body: ChatRequest): ChatStreamHandle {
  const controller = new AbortController();
  return { events: runStream(body, controller.signal), controller };
}

async function* runStream(body: ChatRequest, signal: AbortSignal): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "X-Tenant-Id": TENANT_ID,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      const errBody: unknown = await res.json();
      if (errBody && typeof errBody === "object" && "detail" in errBody) {
        detail = String((errBody as { detail: unknown }).detail);
      }
    } catch {
      // no JSON body to report
    }
    throw new Error(`${res.status} /chat/stream: ${detail}`);
  }

  const reader = res.body.getReader();
  try {
    yield* parseEventStream(reader);
  } finally {
    reader.releaseLock();
  }
}

async function* parseEventStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
): AsyncGenerator<StreamEvent> {
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseFrame(frame);
      if (event) yield event;
    }
  }

  // A frame with no trailing blank line (e.g. the connection closed right
  // after the last event) is still worth parsing rather than dropping.
  if (buffer.trim()) {
    const event = parseFrame(buffer);
    if (event) yield event;
  }
}

/** Strips the field name and the single optional leading space the SSE spec
 * allows — not all whitespace, which would corrupt multi-line text payloads. */
function fieldValue(line: string, field: string): string {
  const value = line.slice(field.length);
  return value.startsWith(" ") ? value.slice(1) : value;
}

function parseFrame(rawFrame: string): StreamEvent | null {
  let eventName: string | null = null;
  const dataLines: string[] = [];

  for (const rawLine of rawFrame.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (line.startsWith(":")) continue; // heartbeat/comment, not data
    if (line.startsWith("event:")) {
      eventName = fieldValue(line, "event:").trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(fieldValue(line, "data:"));
    }
  }

  if (dataLines.length === 0) return null;

  // Per the SSE spec multiple `data:` lines rejoin with a newline.
  const body = dataLines.join("\n");
  if (!body.trim()) return null;

  let payload: { type?: string; data?: unknown };
  try {
    payload = JSON.parse(body) as { type?: string; data?: unknown };
  } catch {
    // One corrupt frame must not take down the run. Drop it and keep reading —
    // the modules still reporting are worth more than a clean failure.
    console.warn("Discarded an unparseable SSE frame", { event: eventName });
    return null;
  }

  const type = payload.type ?? eventName;
  if (!type) return null;

  return { type, data: payload.data } as StreamEvent;
}
