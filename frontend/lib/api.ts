import type { IdentifyResponse, Table } from "./types";

const BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

// Always include ngrok's bypass header. Harmless on localhost / non-ngrok hosts;
// required for ngrok free tier to return our JSON instead of its HTML interstitial.
const DEFAULT_HEADERS: HeadersInit = { "ngrok-skip-browser-warning": "true" };

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  for (const [k, v] of Object.entries(DEFAULT_HEADERS)) headers.set(k, v);
  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${text || res.statusText}`);
  }
  return res.json();
}

export async function fetchLocations(): Promise<{ locations: string[]; default: string }> {
  return jsonFetch(`${BASE}/locations`);
}

export async function prewarm(location: string): Promise<{ loaded: boolean; from_cache: boolean }> {
  const fd = new FormData();
  fd.append("location", location);
  return jsonFetch(`${BASE}/prewarm`, { method: "POST", body: fd });
}

export async function identify(blob: Blob, location: string): Promise<IdentifyResponse> {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  fd.append("location", location);
  return jsonFetch(`${BASE}/identify`, { method: "POST", body: fd });
}

export type CreateTableInput = {
  table_name: string;
  restaurant: string;
  city: string;
  ayce_price_per_person: number;
  tax_included: boolean;
  tip_percent: number;
  host_name: string;
};

export async function createTable(input: CreateTableInput): Promise<{ table: Table; participant_id: string }> {
  return jsonFetch(`${BASE}/tables`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export async function joinTable(code: string, name: string): Promise<{ table: Table; participant_id: string }> {
  return jsonFetch(`${BASE}/tables/${code}/join`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function getTable(code: string): Promise<Table> {
  return jsonFetch(`${BASE}/tables/${code}`);
}

export async function addTableCapture(
  code: string,
  body: { participant_id: string; total: number; counts: unknown[]; pricing: unknown },
): Promise<Table> {
  return jsonFetch(`${BASE}/tables/${code}/captures`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function finishTable(code: string): Promise<Table> {
  return jsonFetch(`${BASE}/tables/${code}/finish`, { method: "POST" });
}
