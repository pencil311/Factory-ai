import { motion, useReducedMotion } from "framer-motion";
import { ArrowUp, Mic, MicOff, Square } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { useSpeechRecognition } from "@/hooks/use-speech-recognition";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

/** BCP-47 for recognition; the two-letter prefix is what the backend wants. */
export const LANGUAGES = [
  { code: "en-US", label: "English" },
  { code: "de-DE", label: "Deutsch" },
  { code: "fr-FR", label: "Français" },
  { code: "es-ES", label: "Español" },
  { code: "it-IT", label: "Italiano" },
  { code: "pl-PL", label: "Polski" },
  { code: "pt-PT", label: "Português" },
  { code: "tr-TR", label: "Türkçe" },
] as const;

export function Composer({
  streaming,
  language,
  onLanguageChange,
  onSend,
  onStop,
}: {
  streaming: boolean;
  language: string;
  onLanguageChange: (code: string) => void;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const [value, setValue] = useState("");
  const reduceMotion = useReducedMotion();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Final speech segments append to whatever is already typed, so dictation
  // and typing compose instead of overwriting each other.
  const commit = useCallback((text: string) => {
    setValue((current) => (current ? `${current.trimEnd()} ${text}` : text));
  }, []);

  const speech = useSpeechRecognition(language, commit);

  const send = () => {
    if (!value.trim() || streaming) return;
    if (speech.listening) speech.stop();
    onSend(value);
    setValue("");
  };

  return (
    <div className="border-t border-border bg-background px-4 py-3">
      <div className="mx-auto max-w-3xl">
        {speech.listening && (
          <motion.div
            initial={reduceMotion ? false : { opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.18, ease: [0.4, 0, 0.2, 1] }}
            className="mb-2 flex items-start gap-2 rounded-md border border-primary/40 bg-primary/5 px-3 py-2"
            aria-live="polite"
          >
            <motion.span
              className="mt-1.5 size-2 shrink-0 rounded-full bg-primary"
              animate={reduceMotion ? undefined : { opacity: [1, 0.25, 1] }}
              transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
            />
            <p className="min-w-0 flex-1 text-sm leading-relaxed">
              <span className="font-medium text-primary">Listening</span>
              {speech.interim && <span className="text-muted-foreground"> — {speech.interim}</span>}
            </p>
          </motion.div>
        )}

        {speech.error && (
          <p className="mb-2 text-xs leading-relaxed text-warning" role="status">
            {speech.error}
          </p>
        )}

        <div className="flex items-end gap-2">
          {/* Deliberately oversized: a gloved thumb needs a target this big. */}
          <button
            type="button"
            onClick={speech.listening ? speech.stop : speech.start}
            disabled={!speech.supported}
            aria-label={
              !speech.supported
                ? "Dictation is not supported in this browser"
                : speech.listening
                  ? "Stop dictating"
                  : "Dictate a message"
            }
            aria-pressed={speech.listening}
            title={
              speech.supported ? undefined : "Dictation needs a Chromium-based browser or Safari."
            }
            className={cn(
              "relative grid size-14 shrink-0 place-items-center rounded-md border transition-colors duration-200",
              speech.listening
                ? "border-primary bg-primary text-primary-foreground"
                : "border-border bg-card text-foreground hover:bg-muted",
              !speech.supported && "cursor-not-allowed opacity-40 hover:bg-card",
            )}
          >
            {speech.supported ? <Mic className="size-6" /> : <MicOff className="size-6" />}
            {speech.listening && !reduceMotion && (
              <motion.span
                aria-hidden
                className="absolute inset-0 rounded-md ring-2 ring-primary"
                animate={{ opacity: [0.8, 0, 0.8] }}
                transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
              />
            )}
          </button>

          <div className="min-w-0 flex-1 rounded-md border border-border bg-card focus-within:border-primary">
            <Textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Describe the problem, or name a machine and an error code…"
              rows={2}
              className="min-h-[56px] resize-none border-0 bg-transparent text-sm focus-visible:ring-0"
            />
            <div className="flex items-center justify-between gap-2 border-t border-border/60 px-2 py-1.5">
              <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                <span className="sr-only">Dictation and answer language</span>
                <select
                  value={language}
                  onChange={(e) => onLanguageChange(e.target.value)}
                  className="cursor-pointer rounded-sm bg-transparent py-0.5 pr-1 text-[11px] outline-none hover:text-foreground focus-visible:ring-1 focus-visible:ring-ring"
                >
                  {LANGUAGES.map((l) => (
                    <option
                      key={l.code}
                      value={l.code}
                      className="bg-popover text-popover-foreground"
                    >
                      {l.label}
                    </option>
                  ))}
                </select>
              </label>
              <span className="text-[11px] text-muted-foreground">
                Enter to send · Shift+Enter for a new line
              </span>
            </div>
          </div>

          {streaming ? (
            <Button
              type="button"
              onClick={onStop}
              variant="outline"
              className="size-14 shrink-0 p-0"
              aria-label="Stop the run"
            >
              <Square className="size-5" />
            </Button>
          ) : (
            <Button
              type="button"
              onClick={send}
              disabled={!value.trim()}
              className="size-14 shrink-0 p-0"
              aria-label="Send"
            >
              <ArrowUp className="size-5" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
