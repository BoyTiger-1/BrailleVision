const $ = (id) => document.getElementById(id);

const video = $("video");
const placeholder = $("placeholder");
const captureCanvas = $("captureCanvas");
const debugImage = $("debugImage");
const recognizedText = $("recognizedText");
const statConf = $("statConf");
const statDots = $("statDots");
const statCells = $("statCells");
const hintEl = $("hint");
const errorEl = $("error");
const batchResults = $("batchResults");

const btnCamera = $("btnCamera");
const btnScan = $("btnScan");
const btnLive = $("btnLive");
const btnStop = $("btnStop");
const btnSpeak = $("btnSpeak");
const btnStopSpeak = $("btnStopSpeak");
const fileInput = $("fileInput");

let stream = null;
let liveTimer = null;
let scanning = false;
let lastSpoken = "";

const LIVE_MS = 900;

function apiBase() {
  const v = $("apiUrl").value.trim();
  if (v) return v.replace(/\/$/, "");
  if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
    return `${location.protocol}//${location.hostname}:8000`;
  }
  return "";
}

function voiceOn() {
  return $("voiceGuide").checked;
}

function speak(text, force = false) {
  if (!text?.trim() || !voiceOn()) return;
  if (!force && text === lastSpoken) return;
  lastSpoken = text;
  if (!("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  speechSynthesis.speak(u);
}

function announce(text) {
  lastSpoken = "";
  speak(text, true);
}

function stopSpeech() {
  speechSynthesis.cancel();
}

function setError(msg) {
  if (!msg) {
    errorEl.hidden = true;
    errorEl.textContent = "";
    return;
  }
  errorEl.hidden = false;
  errorEl.textContent = msg;
}

function applyResult(data, label = "") {
  const text = data.text || "—";
  if (label) {
    batchResults.hidden = false;
    const div = document.createElement("div");
    div.className = "batch-item";
    div.innerHTML = `<strong>${label}</strong>: ${text} <span>(${(data.confidence * 100).toFixed(0)}%)</span>`;
    batchResults.appendChild(div);
  } else {
    recognizedText.textContent = text;
    statConf.textContent = `${Math.round((data.confidence || 0) * 100)}%`;
    statDots.textContent = String(data.dot_count ?? "—");
    statCells.textContent = String(data.cell_count ?? "—");
    hintEl.textContent = data.alignment_hint || "";
    hintEl.className = data.dot_count < 3 ? "hint warn" : "hint";
    btnSpeak.disabled = !data.text;
    if (data.text) speak(data.text);
  }

  if ($("showDebug").checked && data.debug_image) {
    debugImage.src = `data:image/jpeg;base64,${data.debug_image}`;
    debugImage.hidden = false;
  } else if (!label) {
    debugImage.hidden = true;
  }
}

async function scanBase64(dataUrl) {
  const base = apiBase();
  if (!base) throw new Error("Set API URL to your running BrailleVision server.");
  const res = await fetch(`${base}/scan/base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl, include_debug: $("showDebug").checked }),
  });
  if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
  return res.json();
}

async function scanBatchBase64(urls) {
  const base = apiBase();
  const res = await fetch(`${base}/scan/batch/base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images: urls, include_debug: false }),
  });
  if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
  return res.json();
}

function captureFrame() {
  if (!video.videoWidth) return null;
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(video, 0, 0);
  return captureCanvas.toDataURL("image/jpeg", 0.88);
}

async function runScan() {
  if (scanning) return;
  const frame = captureFrame();
  if (!frame) return;
  scanning = true;
  btnScan.disabled = true;
  setError("");
  batchResults.hidden = true;
  batchResults.innerHTML = "";
  try {
    const data = await scanBase64(frame);
    applyResult(data);
  } catch (e) {
    setError(e.message || "Scan failed");
    if (voiceOn()) announce("Scan failed. Check that the API server is running.");
  } finally {
    scanning = false;
    btnScan.disabled = !stream;
  }
}

async function startCamera() {
  stopSpeech();
  setError("");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    video.hidden = false;
    placeholder.hidden = true;
    btnCamera.disabled = true;
    btnScan.disabled = false;
    btnLive.disabled = false;
    btnStop.disabled = false;
    if (voiceOn()) {
      announce("Camera on. Point at Braille and tap Scan now or Live scan.");
    }
  } catch {
    setError("Camera permission denied or unavailable.");
    if (voiceOn()) announce("Could not start camera.");
  }
}

function stopCamera() {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  btnLive.setAttribute("aria-pressed", "false");
  btnLive.textContent = "Live scan";
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  video.hidden = true;
  placeholder.hidden = false;
  btnCamera.disabled = false;
  btnScan.disabled = true;
  btnLive.disabled = true;
  btnStop.disabled = true;
}

function toggleLive() {
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
    btnLive.setAttribute("aria-pressed", "false");
    btnLive.textContent = "Live scan";
    if (voiceOn()) announce("Live scan off");
    return;
  }
  liveTimer = setInterval(() => void runScan(), LIVE_MS);
  btnLive.setAttribute("aria-pressed", "true");
  btnLive.textContent = "Stop live scan";
  if (voiceOn()) announce("Live scan on");
}

function readFilesAsDataUrls(files) {
  return Promise.all(
    [...files].map(
      (file) =>
        new Promise((resolve, reject) => {
          const r = new FileReader();
          r.onload = () => resolve({ name: file.name, url: r.result });
          r.onerror = reject;
          r.readAsDataURL(file);
        })
    )
  );
}

async function onFilesSelected(files) {
  if (!files?.length) return;
  setError("");
  batchResults.innerHTML = "";
  const items = await readFilesAsDataUrls(files);

  if (items.length === 1) {
    try {
      const data = await scanBase64(items[0].url);
      applyResult(data);
    } catch (e) {
      setError(e.message);
    }
    return;
  }

  try {
    const results = await scanBatchBase64(items.map((i) => i.url));
    batchResults.hidden = false;
    recognizedText.textContent = results.map((r) => r.text).join(" ");
    results.forEach((r, i) => applyResult(r, items[i].name));
    const combined = results.map((r) => r.text).filter(Boolean).join(". ");
    if (combined && voiceOn()) speak(combined, true);
  } catch (e) {
    setError(e.message);
  }
}

btnCamera.addEventListener("click", () => void startCamera());
btnScan.addEventListener("click", () => void runScan());
btnLive.addEventListener("click", toggleLive);
btnStop.addEventListener("click", stopCamera);
btnSpeak.addEventListener("click", () => speak(recognizedText.textContent, true));
btnStopSpeak.addEventListener("click", stopSpeech);
fileInput.addEventListener("change", (e) => void onFilesSelected(e.target.files));

$("highContrast").addEventListener("change", (e) => {
  document.body.classList.toggle("high-contrast", e.target.checked);
});
document.body.classList.add("high-contrast");

if (location.port === "8000") {
  $("apiUrl").value = location.origin;
}
