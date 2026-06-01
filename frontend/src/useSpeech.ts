import { useCallback, useRef } from "react";

export function useSpeech() {
  const lastSpoken = useRef("");

  const speak = useCallback((text: string, force = false) => {
    if (!text.trim()) return;
    if (!force && text === lastSpoken.current) return;
    lastSpoken.current = text;
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.95;
    u.pitch = 1;
    window.speechSynthesis.speak(u);
  }, []);

  const announce = useCallback((text: string) => {
    speak(text, true);
  }, [speak]);

  const stop = useCallback(() => {
    window.speechSynthesis?.cancel();
  }, []);

  return { speak, announce, stop };
}
