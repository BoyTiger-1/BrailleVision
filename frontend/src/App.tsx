import { useCallback, useEffect, useRef, useState } from "react";
import { scanFrame, type ScanResult } from "./api";
import { useSpeech } from "./useSpeech";
import "./App.css";

const SCAN_INTERVAL_MS = 900;

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanningRef = useRef(false);

  const [cameraOn, setCameraOn] = useState(false);
  const [liveScan, setLiveScan] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [voiceGuide, setVoiceGuide] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [highContrast, setHighContrast] = useState(true);

  const { speak, announce, stop } = useSpeech();

  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return null;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, w, h);
    return canvas.toDataURL("image/jpeg", 0.85);
  }, []);

  const runScan = useCallback(async () => {
    if (scanningRef.current) return;
    const frame = captureFrame();
    if (!frame) return;
    scanningRef.current = true;
    setLoading(true);
    setError(null);
    try {
      const data = await scanFrame(frame, showDebug);
      setResult(data);
      if (data.text && voiceGuide) {
        speak(data.text);
      }
      if (voiceGuide && data.alignment_hint) {
        const hintKey = data.alignment_hint.slice(0, 40);
        if (data.dot_count === 0 || data.cell_count === 0) {
          speak(hintKey, true);
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Scan failed";
      setError(msg);
    } finally {
      setLoading(false);
      scanningRef.current = false;
    }
  }, [captureFrame, showDebug, speak, voiceGuide]);

  const startCamera = useCallback(async () => {
    stop();
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }
      setCameraOn(true);
      if (voiceGuide) {
        announce(
          "Camera started. Point at physical Braille. Use Scan now or turn on live scanning."
        );
      }
    } catch {
      setError("Camera access denied or unavailable. Allow camera permission and retry.");
      if (voiceGuide) announce("Camera could not start. Check permissions.");
    }
  }, [announce, stop, voiceGuide]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    setLiveScan(false);
  }, []);

  useEffect(() => {
    if (!liveScan || !cameraOn) return;
    const id = setInterval(() => {
      void runScan();
    }, SCAN_INTERVAL_MS);
    return () => clearInterval(id);
  }, [liveScan, cameraOn, runScan]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const onUpload = async (file: File) => {
    setError(null);
    setLoading(true);
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const dataUrl = reader.result as string;
        const data = await scanFrame(dataUrl, showDebug);
        setResult(data);
        if (voiceGuide && data.text) speak(data.text, true);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload scan failed");
      } finally {
        setLoading(false);
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className={`app ${highContrast ? "high-contrast" : ""}`}>
      <header>
        <h1>BrailleVision</h1>
        <p className="tagline">Scan physical Braille — dots on paper, not Unicode.</p>
      </header>

      <main id="main">
        <section aria-labelledby="camera-heading" className="camera-section">
          <h2 id="camera-heading">Camera</h2>

          <div className="video-wrap" role="img" aria-label="Live camera preview for Braille scanning">
            {!cameraOn && (
              <div className="placeholder">
                <p>Start the camera to scan embossed or handwritten Braille.</p>
              </div>
            )}
            <video ref={videoRef} playsInline muted className={cameraOn ? "active" : ""} />
            <canvas ref={canvasRef} className="sr-only" aria-hidden />
            {cameraOn && <div className="alignment-frame" aria-hidden />}
          </div>

          {result?.debug_image && showDebug && (
            <img
              className="debug-img"
              src={`data:image/jpeg;base64,${result.debug_image}`}
              alt="Debug view showing detected dots"
            />
          )}

          <div className="controls" role="toolbar" aria-label="Scanner controls">
            {!cameraOn ? (
              <button type="button" className="primary" onClick={() => void startCamera()}>
                Start camera
              </button>
            ) : (
              <>
                <button type="button" className="primary" onClick={() => void runScan()} disabled={loading}>
                  {loading ? "Scanning…" : "Scan now"}
                </button>
                <button
                  type="button"
                  aria-pressed={liveScan}
                  onClick={() => {
                    setLiveScan((v) => !v);
                    if (voiceGuide) {
                      announce(liveScan ? "Live scanning off" : "Live scanning on");
                    }
                  }}
                >
                  {liveScan ? "Stop live scan" : "Live scan"}
                </button>
                <button type="button" onClick={stopCamera}>
                  Stop camera
                </button>
              </>
            )}

            <label className="upload-btn">
              Upload image
              <input
                type="file"
                accept="image/*"
                className="sr-only"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void onUpload(f);
                }}
              />
            </label>
          </div>
        </section>

        <section aria-labelledby="output-heading" className="output-section" aria-live="polite" aria-atomic="true">
          <h2 id="output-heading">Recognized text</h2>
          <p className="recognized-text">{result?.text || "—"}</p>
          {result && (
            <dl className="stats">
              <div>
                <dt>Confidence</dt>
                <dd>{Math.round(result.confidence * 100)}%</dd>
              </div>
              <div>
                <dt>Dots</dt>
                <dd>{result.dot_count}</dd>
              </div>
              <div>
                <dt>Cells</dt>
                <dd>{result.cell_count}</dd>
              </div>
            </dl>
          )}
          {result?.alignment_hint && (
            <p className={`hint ${result.dot_count < 3 ? "warn" : "ok"}`} role="status">
              {result.alignment_hint}
            </p>
          )}
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {result?.text && (
            <button
              type="button"
              onClick={() => speak(result.text, true)}
              aria-label="Read recognized text aloud again"
            >
              Read aloud
            </button>
          )}
        </section>

        <section aria-labelledby="settings-heading" className="settings">
          <h2 id="settings-heading">Accessibility</h2>
          <label className="toggle">
            <input
              type="checkbox"
              checked={voiceGuide}
              onChange={(e) => setVoiceGuide(e.target.checked)}
            />
            Voice guidance &amp; auto read
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={highContrast}
              onChange={(e) => setHighContrast(e.target.checked)}
            />
            High contrast
          </label>
          <label className="toggle">
            <input type="checkbox" checked={showDebug} onChange={(e) => setShowDebug(e.target.checked)} />
            Show detection overlay
          </label>
          <button type="button" onClick={() => stop()} aria-label="Stop speech">
            Stop speech
          </button>
        </section>
      </main>

      <footer>
        <p>BrailleVision Hackathon 2026 — physical dot detection via OpenCV</p>
      </footer>
    </div>
  );
}
