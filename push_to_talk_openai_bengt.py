import asyncio
import os
import re
import tempfile
import time
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

SWEDISH_BENGT_ID = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_svSE_Bengt"

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024

SYSTEM_PROMPT = (
    "Du ar en hjalpsam svensk assistent. "
    "Svara pa svenska. Var tydlig och konkret. "
    "Om frågan galler aktuell information, använd webben nar det behovs. "
    "Svara normalt kort (3-6 meningar) om inte anvandaren ber om ett langt svar."
)

EDGE_VOICES = [
    "sv-SE-MattiasNeural",
    "sv-SE-SofieNeural",
    "sv-SE-HeddaNeural",
    "sv-SE-ErikNeural",
]
EDGE_RATE = "+30%"
EDGE_PITCH = "+2Hz"


def make_tts_engine() -> pyttsx3.Engine:
    engine = pyttsx3.init()
    engine.setProperty("rate", 195)

    try:
        engine.setProperty("voice", SWEDISH_BENGT_ID)
        return engine
    except Exception:
        pass

    for voice in engine.getProperty("voices"):
        voice_id = (voice.id or "").lower()
        voice_name = (voice.name or "").lower()
        if "bengt" in voice_name or "sv" in voice_id or "svenska" in voice_name:
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
    chunks: list[str] = []
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


def speak(engine: pyttsx3.Engine, text: str) -> None:
    if not text:
        return

    chunks = _split_for_tts(text)
    if not chunks:
        return

    if EDGE_TTS_AVAILABLE:
        for voice in EDGE_VOICES:
            try:
                for chunk in chunks:
                    tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name)
                    try:
                        asyncio.run(_edge_save_audio(chunk, voice, str(tmp)))
                        audio, samplerate = sf.read(str(tmp), dtype="float32")
                        sd.play(audio, samplerate)
                        sd.wait()
                    finally:
                        tmp.unlink(missing_ok=True)
                return
            except Exception as e:
                print(f"Edge TTS misslyckades med {voice}: {e}")
                continue

        print("Edge TTS misslyckades for alla roster. Faller tillbaka till lokal rost.")

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
    print("\nHall inne ENTER och prata. Slapp ENTER for att stoppa. ESC avslutar.")
    if not wait_for_enter_hold():
        return None

    frames: list[np.ndarray] = []
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


def transcribe_audio(client: OpenAI, wav_path: str) -> str:
    with open(wav_path, "rb") as audio_file:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="sv",
        )
    return (resp.text or "").strip()


def ask_openai(client: OpenAI, user_text: str) -> str:
    """
    Forsok ChatGPT-likt svar med Responses API + web search.
    Fallback till chat.completions om responses/tooling inte fungerar.
    """
    tool_types = ["web_search", "web_search_preview"]
    for tool_type in tool_types:
        try:
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY saknas i .env")

    client = OpenAI(api_key=api_key)
    engine = make_tts_engine()

    print("Push-to-talk redo.")
    print("Kor terminalen som Administrator om ENTER-hold inte registreras.")
    if EDGE_TTS_AVAILABLE:
        print(f"Edge TTS aktivt. Prioriterade roster: {', '.join(EDGE_VOICES)}")
    else:
        print("Edge TTS saknas. Kor lokal Bengt-rost i stallet.")

    while True:
        wav_path = record_while_enter_held()
        if wav_path is None:
            print("Avslutar.")
            break
        if wav_path == "":
            continue

        try:
            user_text = transcribe_audio(client, wav_path)
            if not user_text:
                print("Horde inget.")
                speak(engine, "Jag horde inget. Forsok igen.")
                continue

            print(f"Du sa: {user_text}")
            answer = ask_openai(client, user_text)
            print(f"Assistent: {answer}")
            speak(engine, answer or "Jag fick inget svar just nu.")
        except Exception as e:
            print(f"Fel: {e}")
            speak(engine, "Nagot gick fel. Prova igen.")
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
