import asyncio
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import keyboard
import numpy as np
import pyttsx3
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

try:
    from edge_tts import Communicate
    EDGE_TTS_AVAILABLE = True
except Exception:
    Communicate = None
    EDGE_TTS_AVAILABLE = False

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024

EDGE_RATE = os.getenv("EDGE_TTS_RATE", "+15%")
EDGE_PITCH = os.getenv("EDGE_TTS_PITCH", "+0Hz")
EDGE_RETRIES = 2


@dataclass
class VoiceProfile:
    key: str
    label: str
    language: str
    primary_edge_voice: str


VOICE_PROFILES = [
    VoiceProfile("1", "Svenska - Hedda (Edge)", "sv", "sv-SE-HeddaNeural"),
    VoiceProfile("2", "Svenska - Sofie (Edge)", "sv", "sv-SE-SofieNeural"),
    VoiceProfile("3", "English - Jenny (Edge)", "en", "en-US-JennyNeural"),
]

EDGE_FALLBACKS = {
    "sv": ["sv-SE-HeddaNeural", "sv-SE-SofieNeural", "sv-SE-MattiasNeural", "sv-SE-ErikNeural"],
    "en": ["en-US-JennyNeural", "en-US-AriaNeural", "en-US-GuyNeural"],
}


def select_profile() -> VoiceProfile:
    print("\nVälj röst + språk:")
    for profile in VOICE_PROFILES:
        print(f"  {profile.key}. {profile.label}")

    while True:
        choice = input("Ditt val (1-3): ").strip()
        for profile in VOICE_PROFILES:
            if profile.key == choice:
                return profile
        print("Ogiltigt val. Skriv 1, 2 eller 3.")


def system_prompt_for(language: str) -> str:
    if language == "sv":
        return (
            "Du är en hjälpsam svensk assistent. "
            "Svara på svenska. Var tydlig och konkret. "
            "Om frågan gäller aktuell information, använd webben när det behövs. "
            "Svara normalt kort (3-6 meningar) om inte användaren ber om ett långt svar."
        )

    return (
        "You are a helpful assistant. "
        "Reply in English with clear, practical answers. "
        "If the user asks for current information, use web search when needed. "
        "Keep answers normally concise unless the user asks for depth."
    )


def make_tts_engine(language: str) -> pyttsx3.Engine:
    engine = pyttsx3.init()
    engine.setProperty("rate", 195)

    for voice in engine.getProperty("voices"):
        voice_id = (voice.id or "").lower()
        voice_name = (voice.name or "").lower()
        if language == "sv" and ("sv" in voice_id or "svenska" in voice_name):
            engine.setProperty("voice", voice.id)
            break
        if language == "en" and ("en" in voice_id or "english" in voice_name):
            engine.setProperty("voice", voice.id)
            break

    return engine


def _speak_local(engine: pyttsx3.Engine, text: str) -> None:
    if not text:
        return
    engine.say(text)
    engine.runAndWait()


async def _edge_save_audio(text: str, voice: str, out_path: str) -> None:
    if not EDGE_TTS_AVAILABLE:
        raise RuntimeError("edge-tts saknas")
    communicate = Communicate(text=text, voice=voice, rate=EDGE_RATE, pitch=EDGE_PITCH)
    await communicate.save(out_path)


def _clean_text_for_tts(text: str) -> str:
    cleaned = text
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{2,}", ". ", cleaned)
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("`", "")
    cleaned = cleaned.replace("#", "")
    cleaned = cleaned.replace("* ", "")
    cleaned = cleaned.replace("- ", "")
    cleaned = cleaned.replace("\\(", "")
    cleaned = cleaned.replace("\\)", "")
    cleaned = cleaned.replace("\\[", "")
    cleaned = cleaned.replace("\\]", "")
    cleaned = re.sub(r"\$([^$]+)\$", r"\1", cleaned)
    cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_for_tts(text: str, max_chars: int = 260) -> list[str]:
    text = _clean_text_for_tts(text)
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
                chunks.append(part[i : i + max_chars].strip())
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


def _edge_voice_order(profile: VoiceProfile) -> list[str]:
    base = EDGE_FALLBACKS.get(profile.language, [])
    return list(dict.fromkeys([profile.primary_edge_voice] + base))


def _speak_with_edge(chunks: list[str], profile: VoiceProfile) -> bool:
    if not EDGE_TTS_AVAILABLE:
        return False

    for voice in _edge_voice_order(profile):
        voice_ok = True
        for chunk in chunks:
            chunk_ok = False
            last_error = None
            for _ in range(EDGE_RETRIES):
                tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name)
                try:
                    asyncio.run(_edge_save_audio(chunk, voice, str(tmp)))
                    audio, samplerate = sf.read(str(tmp), dtype="float32")
                    sd.play(audio, samplerate)
                    sd.wait()
                    chunk_ok = True
                    break
                except Exception as e:
                    last_error = e
                    time.sleep(0.15)
                finally:
                    tmp.unlink(missing_ok=True)
            if not chunk_ok:
                voice_ok = False
                print(f"Edge TTS misslyckades med {voice}: {last_error}")
                break
        if voice_ok:
            return True

    return False


def speak(engine: pyttsx3.Engine, text: str, profile: VoiceProfile) -> None:
    chunks = _split_for_tts(text)
    if not chunks:
        return

    if _speak_with_edge(chunks, profile):
        return

    print("TTS fallback till lokal röst.")
    try:
        for chunk in chunks:
            _speak_local(engine, chunk)
    except Exception as e:
        print(f"Lokal TTS misslyckades: {e}")


def wait_for_enter_hold() -> bool:
    while True:
        if keyboard.is_pressed("esc"):
            return False
        if keyboard.is_pressed("enter"):
            return True
        time.sleep(0.02)


def record_while_enter_held() -> str | None:
    print("\nHåll inne ENTER och prata. Släpp ENTER för att stoppa. ESC avslutar.")
    if not wait_for_enter_hold():
        return None

    frames = []
    print("Spelar in...")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCK_SIZE,
        ) as stream:
            while keyboard.is_pressed("enter"):
                chunk, overflowed = stream.read(BLOCK_SIZE)
                if overflowed:
                    print("Varning: audio overflow.")
                frames.append(chunk.copy())
                time.sleep(0.001)
    except Exception as e:
        print(f"Kunde inte spela in ljud: {e}")
        return ""

    if not frames:
        return ""

    audio = np.concatenate(frames, axis=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    sf.write(tmp.name, audio, SAMPLE_RATE)
    return tmp.name


def transcribe_audio(client: OpenAI, wav_path: str, language: str) -> str:
    with open(wav_path, "rb") as audio_file:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
        )
    return (resp.text or "").strip()


def ask_openai(client: OpenAI, user_text: str, language: str) -> str:
    system_prompt = system_prompt_for(language)
    tool_types = ["web_search", "web_search_preview"]

    for tool_type in tool_types:
        try:
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                tools=[{"type": tool_type}],
            )
            text = (getattr(resp, "output_text", None) or "").strip()
            if text:
                return text
        except Exception:
            continue

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY saknas i .env")

    profile = select_profile()
    client = OpenAI(api_key=api_key)
    engine = make_tts_engine(profile.language)

    print("\nPush-to-talk redo.")
    print("Kör terminalen som Administrator om ENTER-hold inte registreras.")
    print(f"Vald profil: {profile.label}")

    while True:
        wav_path = record_while_enter_held()
        if wav_path is None:
            print("Avslutar.")
            break
        if wav_path == "":
            continue

        try:
            user_text = transcribe_audio(client, wav_path, profile.language)
            if not user_text:
                print("Hörde inget.")
                fallback = "Jag hörde inget. Försök igen." if profile.language == "sv" else "I did not hear anything. Please try again."
                speak(engine, fallback, profile)
                continue

            print(f"Du sa: {user_text}")
            answer = ask_openai(client, user_text, profile.language)
            print(f"Assistent: {answer}")
            fallback = "Jag fick inget svar just nu." if profile.language == "sv" else "I could not get an answer right now."
            speak(engine, answer or fallback, profile)
        except Exception as e:
            print(f"Fel: {e}")
            fallback = "Något gick fel. Prova igen." if profile.language == "sv" else "Something went wrong. Please try again."
            speak(engine, fallback, profile)
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
