import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Minimal typings for the Web Speech API, which is not in lib.dom and is
 * still vendor-prefixed in Chromium.
 */
interface SpeechRecognitionAlternative {
  transcript: string;
}
interface SpeechRecognitionResult {
  readonly length: number;
  readonly isFinal: boolean;
  [index: number]: SpeechRecognitionAlternative;
}
interface SpeechRecognitionResultList {
  readonly length: number;
  [index: number]: SpeechRecognitionResult;
}
interface SpeechRecognitionEventLike extends Event {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
}
interface SpeechRecognitionErrorEventLike extends Event {
  readonly error: string;
}
interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function getRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

const ERROR_MESSAGES: Record<string, string> = {
  "not-allowed": "Microphone access was blocked. Allow it in your browser settings to dictate.",
  "service-not-allowed": "Microphone access was blocked by your browser or device policy.",
  "audio-capture": "No microphone was found. Check that one is connected.",
  network: "Speech recognition needs a network connection and could not reach the service.",
  "no-speech": "Nothing was heard. Try again, closer to the microphone.",
};

export interface UseSpeechRecognition {
  /** False when the browser has no Web Speech API — callers must degrade. */
  supported: boolean;
  listening: boolean;
  /** Words recognised but not yet final; replaced as the utterance firms up. */
  interim: string;
  error: string | null;
  start: () => void;
  stop: () => void;
}

/**
 * Dictation with live interim results.
 *
 * Final segments are handed to `onCommit` as they settle rather than being
 * held in state, so the composer owns the committed text and the user can
 * edit it while still dictating.
 */
export function useSpeechRecognition(
  lang: string,
  onCommit: (text: string) => void,
): UseSpeechRecognition {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const commitRef = useRef(onCommit);
  commitRef.current = onCommit;

  // Detected in an effect, not during render: `window` does not exist during
  // SSR and the two passes must agree.
  useEffect(() => setSupported(getRecognitionCtor() !== null), []);

  useEffect(() => {
    const Ctor = getRecognitionCtor();
    if (!Ctor) return;

    const recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.lang = lang;

    recognition.onresult = (event) => {
      let pending = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = result[0]?.transcript ?? "";
        if (result.isFinal) {
          const trimmed = text.trim();
          if (trimmed) commitRef.current(trimmed);
        } else {
          pending += text;
        }
      }
      setInterim(pending);
    };

    recognition.onerror = (event) => {
      setError(ERROR_MESSAGES[event.error] ?? "Dictation stopped unexpectedly.");
      setListening(false);
      setInterim("");
    };

    recognition.onend = () => {
      setListening(false);
      setInterim("");
    };

    recognitionRef.current = recognition;
    return () => {
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      recognition.abort();
      recognitionRef.current = null;
    };
  }, [lang]);

  const start = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    setError(null);
    setInterim("");
    try {
      recognition.start();
      setListening(true);
    } catch {
      // start() throws if already running; treat as already listening.
      setListening(true);
    }
  }, []);

  const stop = useCallback(() => {
    recognitionRef.current?.stop();
    setListening(false);
    setInterim("");
  }, []);

  return { supported, listening, interim, error, start, stop };
}
