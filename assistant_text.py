import os
import re
import sys
import unicodedata
from pathlib import Path


def ensure_tk_runtime_paths() -> None:
    if os.getenv("TCL_LIBRARY") and os.getenv("TK_LIBRARY"):
        return

    candidates = []
    if sys.base_prefix:
        candidates.append(Path(sys.base_prefix) / "tcl")
    candidates.append(Path(sys.executable).resolve().parent.parent / "tcl")

    for base in candidates:
        tcl = base / "tcl8.6" / "init.tcl"
        tk_file = base / "tk8.6" / "tk.tcl"
        if tcl.exists() and tk_file.exists():
            os.environ["TCL_LIBRARY"] = str(tcl.parent)
            os.environ["TK_LIBRARY"] = str(tk_file.parent)
            return


def system_prompt_for(language: str) -> str:
    if language == "sv":
        return (
            "Du ar en hjalpsam svensk assistent. "
            "Svara pa svenska. Var tydlig och konkret. "
            "Om fragan galler aktuell information, anvand webben nar det behovs. "
            "Svara normalt kort (3-6 meningar) om inte anvandaren ber om ett langt svar."
        )
    return (
        "You are a helpful assistant. "
        "Reply in English with clear, practical answers. "
        "If the user asks for current information, use web search when needed. "
        "Keep answers normally concise unless the user asks for depth."
    )


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def match_tokens(text: str) -> set[str]:
    low = normalize_text(text)
    base_tokens = [t for t in re.findall(r"[a-z0-9]+", low) if t]
    out: set[str] = set()
    for tok in base_tokens:
        out.add(tok)
        if len(tok) > 4 and tok.endswith("s"):
            out.add(tok[:-1])
        if len(tok) > 4 and tok.endswith("n"):
            out.add(tok[:-1])
        if len(tok) > 5 and tok.endswith("en"):
            out.add(tok[:-2])
        if len(tok) > 5 and tok.endswith("et"):
            out.add(tok[:-2])
    return out


def clean_text_for_tts(text: str, short_pauses: bool, reduce_comma_pauses: bool) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{2,}", ". ", cleaned)
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("**", "").replace("`", "").replace("#", "")
    cleaned = cleaned.replace("* ", "").replace("- ", "")
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("\\[", "").replace("\\]", "")
    cleaned = re.sub(r"\$([^$]+)\$", r"\1", cleaned)
    cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
    if short_pauses:
        cleaned = cleaned.replace(";", " ")
        cleaned = cleaned.replace(":", " ")
        cleaned = re.sub(r"[.!?]\s+", " ", cleaned)
    if reduce_comma_pauses:
        cleaned = re.sub(r",\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_for_tts(text: str, short_pauses: bool, reduce_comma_pauses: bool, max_chars: int = 260) -> list[str]:
    text = clean_text_for_tts(text, short_pauses=short_pauses, reduce_comma_pauses=reduce_comma_pauses)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for part in parts:
        if not part:
            continue
        if len(part) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(part), max_chars):
                chunks.append(part[i:i + max_chars].strip())
            continue
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = part
    if current:
        chunks.append(current.strip())
    return chunks
