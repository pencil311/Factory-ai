"""Which language the operator wrote in, and what that does (and does not) change.

An operator on a line in Monterrey types in Spanish; one in Stuttgart types in
German. The prose we write back should be in their language. Nothing else
should change.

That distinction is the whole point of this module:

* **Prose is translated.** The composer is told the target language and writes
  the narrative in it.
* **Structured data is not.** Part numbers, component ids, machine ids, error
  codes, enum values and statuses are identifiers, not words. ``BEARING_WEAR``
  is a key in a fault taxonomy; ``SKF-6310-2RS1`` is what is stamped on the
  box in the storeroom; ``CV-201`` is what is painted on the machine. A
  translated identifier is a wrong identifier, and a technician who goes to
  the shelf looking for a part number that does not exist is worse off than
  one who read an English word. These pass through untouched, always.
* **The template fallback stays English.** When no LLM is available the
  deterministic template renders, and it renders in English. We do not
  machine-translate it — a mistranslated safety instruction is a hazard, and a
  template is exactly the content most likely to be safety-critical. The
  result is marked ``language_fallback=true`` instead, so the reader is told
  they are getting English rather than being quietly handed it.

Detection itself has one non-obvious wrinkle. Maintenance messages are dense
with identifiers — "CV-201 E-4471 SKF-6310-2RS1 vibration 8.2mm/s" is mostly
not language at all. Feeding that to a statistical detector produces noise, so
:func:`linguistic_text` strips the identifiers first and detection declines to
guess when too little natural language is left.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

#: Used when nothing better is known. Also the language every template is in.
DEFAULT_LANGUAGE = "en"

#: Below this many letters of *natural language* (identifiers removed) a
#: statistical detector is guessing, so we do not ask it.
MIN_DETECTABLE_CHARS = 12

#: Confidence below which a detector's answer is discarded.
MIN_DETECTION_CONFIDENCE = 0.55


class LanguageSource(str, Enum):
    """Where the language came from. Surfaced so a caller can judge it."""

    #: The client told us (request body ``language``).
    declared = "DECLARED"
    #: Inferred from the message text.
    detected = "DETECTED"
    #: Nothing usable to go on; fell back to :data:`DEFAULT_LANGUAGE`.
    default = "DEFAULT"


class LanguageDetection(BaseModel):
    """The outcome of asking "what language is this?"."""

    model_config = ConfigDict(use_enum_values=True)

    language: str = DEFAULT_LANGUAGE
    confidence: float = 0.0
    source: LanguageSource = LanguageSource.default
    #: Plain-English explanation, kept for logs and debugging.
    reason: str = ""

    @property
    def is_english(self) -> bool:
        return base_language(self.language) == DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# Identifier stripping
# ---------------------------------------------------------------------------
#: Token shapes that are identifiers rather than words. Order matters only in
#: that each is applied independently; overlapping matches are all removed.
_IDENTIFIER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Part numbers and component ids: SKF-6310-2RS1, NSK-7014A5-P4, CV-201-BRG-D
    re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)+\b"),
    # Enum values and statuses: BEARING_WEAR, REPAIR_NOW, OUT_OF_STOCK
    re.compile(r"\b[A-Z]{2,}(?:_[A-Z0-9]+)+\b"),
    # Machine designators written loosely: "CV 201", "MC110"
    re.compile(r"\b[A-Za-z]{2,4}[- ]?\d{2,4}\b"),
    # Bare measurements and codes: 8.2mm/s, 47.5°C, E4471
    re.compile(r"\b[A-Za-z]?\d+(?:[.,]\d+)?\s?[A-Za-z/°%]{0,6}\b"),
)


def linguistic_text(message: str) -> str:
    """``message`` with identifier-shaped tokens removed.

    What remains is the part a language detector can reason about. Used for
    detection only — the original message is what actually gets processed.
    """
    text = message or ""
    for pattern in _IDENTIFIER_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _letter_count(text: str) -> int:
    return sum(1 for character in text if character.isalpha())


def base_language(code: Optional[str]) -> str:
    """Normalise ``de-DE``/``DE_de``/``  de  `` to ``de``."""
    if not code:
        return DEFAULT_LANGUAGE
    cleaned = str(code).strip().lower().replace("_", "-")
    if not cleaned:
        return DEFAULT_LANGUAGE
    return cleaned.split("-", 1)[0]


# ---------------------------------------------------------------------------
# Deterministic fallback detector
# ---------------------------------------------------------------------------
#: Scripts that identify a language on sight, no word list needed.
_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("ja", ((0x3040, 0x309F), (0x30A0, 0x30FF))),  # kana
    ("ko", ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),  # hangul
    ("zh", ((0x4E00, 0x9FFF),)),  # han — after kana/hangul, which co-occur with it
    ("ru", ((0x0400, 0x04FF),)),  # cyrillic
    ("ar", ((0x0600, 0x06FF),)),
    ("he", ((0x0590, 0x05FF),)),
    ("el", ((0x0370, 0x03FF),)),
    ("hi", ((0x0900, 0x097F),)),
    ("th", ((0x0E00, 0x0E7F),)),
)

#: Closed-class words that are cheap and reliable markers for a language.
#: Deliberately small: this is a fallback for when the detector package is not
#: installed, not a replacement for it.
_MARKER_WORDS: dict[str, frozenset[str]] = {
    "de": frozenset(
        "der die das den dem des ein eine einen und ist nicht mit auf für "
        "von zu im am beim wird werden macht hat haben sich sehr kann muss "
        "läuft wieder nach über aus".split()
    ),
    "es": frozenset(
        "el la los las un una unos unas y es no con para por de del en al "
        "que se está están hace hacer tiene tienen muy pero cuando desde "
        "sobre está".split()
    ),
    "fr": frozenset(
        "le la les un une des et est ne pas avec pour par de du dans au aux "
        "que qui se très mais quand depuis sur fait faire a ont il elle".split()
    ),
    "it": frozenset(
        "il lo la i gli le un uno una e non con per di del nel al che si "
        "molto ma quando da sopra fa fare ha hanno sta stanno".split()
    ),
    "pt": frozenset(
        "o a os as um uma uns umas e não com para por de do da no na que se "
        "muito mas quando desde sobre faz fazer tem têm está estão".split()
    ),
    "nl": frozenset(
        "de het een en is niet met voor van naar op in dat die maar zeer "
        "wordt worden heeft hebben zich erg wanneer over".split()
    ),
    "sv": frozenset(
        "en ett och är inte med för av till på i som men mycket blir har "
        "sig när över gör göra".split()
    ),
    "pl": frozenset(
        "i nie jest z na do w że się bardzo ale kiedy przez dla od po jak "
        "ma mają robi być oraz".split()
    ),
    "tr": frozenset(
        "bir ve değil ile için den dan bu şu çok ama ne zaman üzerinde "
        "var yok yapıyor oluyor".split()
    ),
    "en": frozenset(
        "the a an and is not with for of to on in that but very when from "
        "about does do has have it its this these are was were".split()
    ),
}


def _detect_by_script(text: str) -> Optional[tuple[str, float]]:
    """Language implied by the writing system, if any."""
    counts: dict[str, int] = {}
    for character in text:
        point = ord(character)
        for language, ranges in _SCRIPT_RANGES:
            if any(low <= point <= high for low, high in ranges):
                counts[language] = counts.get(language, 0) + 1
                break

    if not counts:
        return None

    # Japanese and Korean both mix in Han characters; if kana or hangul appear
    # at all, that is the answer regardless of how much Han is present.
    for language in ("ja", "ko"):
        if counts.get(language):
            return language, 0.95

    language, hits = max(counts.items(), key=lambda item: item[1])
    letters = max(_letter_count(text), 1)
    return language, min(0.95, hits / letters)


def _detect_by_markers(text: str) -> Optional[tuple[str, float]]:
    """Language implied by closed-class marker words."""
    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    if not words:
        return None

    scores = {
        language: sum(1 for word in words if word in markers)
        for language, markers in _MARKER_WORDS.items()
    }
    language, hits = max(scores.items(), key=lambda item: item[1])
    if hits == 0:
        return None

    runner_up = max(
        (score for name, score in scores.items() if name != language), default=0
    )
    # A clear winner is worth more than a narrow one; a tie is worth nothing.
    if hits == runner_up:
        return None
    confidence = min(0.9, 0.5 + 0.1 * (hits - runner_up))
    return language, confidence


def heuristic_detect(text: str) -> Optional[tuple[str, float]]:
    """Script-then-markers detection. Used when ``langdetect`` is unavailable."""
    by_script = _detect_by_script(text)
    if by_script is not None:
        return by_script
    return _detect_by_markers(text)


def _langdetect(text: str) -> Optional[tuple[str, float]]:
    """Ask ``langdetect``, or return ``None`` if it is not installed or unsure."""
    try:
        from langdetect import DetectorFactory, detect_langs  # optional dependency
        from langdetect.lang_detect_exception import LangDetectException
    except Exception:
        return None

    # Without a fixed seed langdetect is non-deterministic on short input, and
    # two identical requests must not produce two different languages.
    DetectorFactory.seed = 0

    try:
        ranked = detect_langs(text)
    except LangDetectException:
        return None
    except Exception:  # pragma: no cover - defensive; the package is optional
        logger.warning("Language detection failed unexpectedly", exc_info=True)
        return None

    if not ranked:
        return None
    best = ranked[0]
    return base_language(best.lang), float(best.prob)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def detect_language(
    message: str, *, declared: Optional[str] = None
) -> LanguageDetection:
    """Decide which language to answer ``message`` in.

    A ``declared`` code from the caller always wins: the client knows the
    operator's UI language, and that is better evidence than inference from a
    two-word fault report.
    """
    if declared:
        code = base_language(declared)
        return LanguageDetection(
            language=code,
            confidence=1.0,
            source=LanguageSource.declared,
            reason=f"Caller declared language '{declared}'.",
        )

    text = linguistic_text(message)
    if _letter_count(text) < MIN_DETECTABLE_CHARS:
        return LanguageDetection(
            language=DEFAULT_LANGUAGE,
            confidence=0.0,
            source=LanguageSource.default,
            reason=(
                "Too little natural language to detect from once identifiers "
                f"were removed ('{text}'); defaulting to {DEFAULT_LANGUAGE}."
            ),
        )

    guess = _langdetect(text) or heuristic_detect(text)
    if guess is None:
        return LanguageDetection(
            language=DEFAULT_LANGUAGE,
            confidence=0.0,
            source=LanguageSource.default,
            reason=f"No detector recognised the text; defaulting to {DEFAULT_LANGUAGE}.",
        )

    language, confidence = guess
    if confidence < MIN_DETECTION_CONFIDENCE:
        return LanguageDetection(
            language=DEFAULT_LANGUAGE,
            confidence=confidence,
            source=LanguageSource.default,
            reason=(
                f"Detected '{language}' with confidence {confidence:.2f}, below "
                f"the {MIN_DETECTION_CONFIDENCE} threshold; defaulting to "
                f"{DEFAULT_LANGUAGE}."
            ),
        )

    return LanguageDetection(
        language=language,
        confidence=confidence,
        source=LanguageSource.detected,
        reason=f"Detected '{language}' with confidence {confidence:.2f}.",
    )


# ---------------------------------------------------------------------------
# Naming, for the composer prompt
# ---------------------------------------------------------------------------
LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def language_name(code: Optional[str]) -> str:
    """Human-readable name for an ISO 639-1 code, for use in a prompt."""
    normalised = base_language(code)
    return LANGUAGE_NAMES.get(normalised, normalised)


#: Appended to the composer prompt whenever the answer is not in English. The
#: exclusion list is the contract from this module's docstring, spelled out for
#: a model rather than for a reader.
STRUCTURED_DATA_RULE = (
    "Structured identifiers are language-neutral and MUST be reproduced exactly "
    "as they appear in the brief, never translated, transliterated or "
    "reformatted: part numbers, component ids, machine ids, model names, "
    "sensor names, error codes, enum values and status codes (for example "
    "CV-201, SKF-6310-2RS1, BEARING_WEAR, REPAIR_NOW, IN_STOCK, CRITICAL). "
    "Translate the prose around them, not them."
)


def composer_language_instruction(code: Optional[str]) -> str:
    """The language directive for the composer, or ``""`` for English."""
    normalised = base_language(code)
    if normalised == DEFAULT_LANGUAGE:
        return ""
    return (
        f"Write the entire answer in {language_name(normalised)} "
        f"({normalised}). The operator wrote in that language and must be "
        f"answered in it.\n{STRUCTURED_DATA_RULE}"
    )
