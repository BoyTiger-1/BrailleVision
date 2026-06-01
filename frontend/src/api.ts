export type ScanResult = {
  text: string;
  confidence: number;
  dot_count: number;
  cell_count: number;
  alignment_hint: string;
  debug_image?: string | null;
  cells: { pattern: number; row: number; col: number; confidence: number }[];
};

const API_BASE =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? "/api" : "http://127.0.0.1:8000");

export async function scanFrame(dataUrl: string, includeDebug = false): Promise<ScanResult> {
  const res = await fetch(`${API_BASE}/scan/base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl, include_debug: includeDebug }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `Scan failed (${res.status})`);
  }
  return res.json();
}
