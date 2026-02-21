import asyncio
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext

import keyboard
import numpy as np
import pyttsx3
import sounddevice as sd
import soundfile as sf
from pynput import mouse as pynput_mouse
from dotenv import load_dotenv, set_key
from openai import OpenAI

from assistant_integrations import AssistantIntegrations
from assistant_text import ensure_tk_runtime_paths, normalize_text, split_for_tts, system_prompt_for

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
EDGE_FALLBACK_PROSODY = [
    (EDGE_RATE, EDGE_PITCH),
    ("+0%", "+0Hz"),
    ("+8%", "+0Hz"),
]
SHORT_PAUSES = os.getenv("TTS_SHORT_PAUSES", "1").strip().lower() in {"1", "true", "yes", "on"}
REDUCE_COMMA_PAUSES = os.getenv("TTS_REDUCE_COMMA_PAUSES", "1").strip().lower() in {"1", "true", "yes", "on"}
TTS_ENABLED_DEFAULT = os.getenv("TTS_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class VoiceProfile:
    label: str
    language: str
    primary_edge_voice: str


VOICE_PROFILES = [
    VoiceProfile("Svenska - Sofie (Edge)", "sv", "sv-SE-SofieNeural"),
    VoiceProfile("Svenska - Hedda (Edge)", "sv", "sv-SE-HeddaNeural"),
    VoiceProfile("English - Jenny (Edge)", "en", "en-US-JennyNeural"),
]

EDGE_FALLBACKS = {
    "sv": ["sv-SE-SofieNeural", "sv-SE-HeddaNeural", "sv-SE-MattiasNeural", "sv-SE-ErikNeural"],
    "en": ["en-US-JennyNeural", "en-US-AriaNeural", "en-US-GuyNeural"],
}

class VoiceAssistantGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Voice Assistant")
        self.root.geometry("1080x760")
        self.root.minsize(980, 680)
        self.root.configure(bg="#0b1120")

        load_dotenv(override=True)
        self.env_path = Path(__file__).resolve().parent / ".env"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY saknas i .env")

        self.client = OpenAI(api_key=api_key)
        self.current_profile = VOICE_PROFILES[0]
        self.tts_engine = self._make_local_tts_engine(self.current_profile.language)
        self.tts_enabled = TTS_ENABLED_DEFAULT
        self.keybind = self._normalize_keybind_name(os.getenv("PTT_KEYBIND", "enter").strip() or "enter")
        self.integrations = AssistantIntegrations(log_fn=self._log, save_env_fn=self._save_env)
        self.input_devices = self._get_input_devices()
        self.output_devices = self._get_output_devices()
        saved_input_name = os.getenv("INPUT_DEVICE_NAME", "").strip()
        saved_output_name = os.getenv("OUTPUT_DEVICE_NAME", "").strip()
        self.input_device_index = self._pick_input_device(saved_input_name)
        self.output_device_index = self._pick_output_device(saved_output_name)
        self.running = False
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.capturing_keybind = False
        self.mouse_buttons_down: set[str] = set()
        self.mouse_listener = None

        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.status_queue: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()

        self._start_input_hooks()
        self._apply_audio_device_defaults()
        self._setup_style()
        self._build_ui()
        if self.integrations.hue_bridge_ip and self.integrations.hue_app_key:
            self._log("system", f"Hue ansluten ({self.integrations.hue_bridge_ip}).")
        else:
            self._log("system", "Hue inte ansluten ännu. Försöker ansluta automatiskt.")
        if self.integrations.ha_url and self.integrations.ha_token and self.integrations.ha_vacuum_entity_id:
            self._log("system", f"Home Assistant vacuum konfigurerad ({self.integrations.ha_vacuum_entity_id}).")
        if self.integrations.spotify_client_id and self.integrations.spotify_client_secret and self.integrations.spotify_refresh_token:
            self._log("system", "Spotify konfigurerad.")
        self._log("system", "Redo. Tryck Start och hall vald push-to-talk-tangent.")
        self.integrations.auto_connect_hue()
        self.integrations.auto_connect_vacuum()
        self.integrations.auto_connect_spotify()
        self.root.after(80, self._drain_queues)

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#0b1120")
        style.configure("Card.TFrame", background="#111827")
        style.configure("CardAlt.TFrame", background="#0f172a")

        style.configure("App.TLabel", background="#0b1120", foreground="#dbe7ff", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#111827", foreground="#dbe7ff", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#0b1120", foreground="#f8fbff", font=("Segoe UI Semibold", 24))
        style.configure("Subtle.TLabel", background="#0b1120", foreground="#8ea3c5", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#111827", foreground="#a8c1e8", font=("Segoe UI Semibold", 10))
        style.configure("Status.TLabel", background="#0f172a", foreground="#7dd3fc", font=("Segoe UI Semibold", 10))

        style.configure("App.TButton", font=("Segoe UI Semibold", 10), padding=(12, 9), borderwidth=0)
        style.map("App.TButton", background=[("active", "#24324a"), ("!disabled", "#1b2538")], foreground=[("!disabled", "#e8f1ff")])

        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10), padding=(14, 10), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#0ea5c8"), ("!disabled", "#06b6d4")], foreground=[("!disabled", "#04131a")])

        style.configure("Danger.TButton", font=("Segoe UI Semibold", 10), padding=(12, 9), borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#b91c1c"), ("!disabled", "#dc2626")], foreground=[("!disabled", "#fff1f2")])

        style.configure("App.TCombobox", padding=8, fieldbackground="#0f172a", background="#0f172a", foreground="#e5edff")
        style.map("App.TCombobox", fieldbackground=[("readonly", "#0f172a")], selectbackground=[("readonly", "#1e293b")], selectforeground=[("readonly", "#e5edff")])

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame", padding=20)
        shell.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(shell, style="App.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text="Voice Assistant", style="Title.TLabel").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Status: idle")
        status_pill = ttk.Frame(header, style="CardAlt.TFrame", padding=(12, 8))
        status_pill.pack(side=tk.RIGHT)
        ttk.Label(status_pill, textvariable=self.status_var, style="Status.TLabel").pack()
        ttk.Label(shell, text="Push-to-talk med OpenAI, Edge TTS, Hue och Spotify-kontroll", style="Subtle.TLabel").pack(anchor="w", pady=(2, 14))

        panel = ttk.Frame(shell, style="Card.TFrame", padding=16)
        panel.pack(fill=tk.X)
        ttk.Label(panel, text="Inställningar", style="Section.TLabel").grid(row=0, column=0, sticky="w", columnspan=6, pady=(0, 10))

        ttk.Label(panel, text="Rostprofil", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        self.profile_var = tk.StringVar(value=self.current_profile.label)
        self.profile_combo = ttk.Combobox(panel, textvariable=self.profile_var, values=[p.label for p in VOICE_PROFILES], state="readonly", width=32, style="App.TCombobox")
        self.profile_combo.grid(row=1, column=1, sticky="ew", padx=(8, 14))
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_changed)

        ttk.Label(panel, text="Push-to-talk", style="Card.TLabel").grid(row=1, column=2, sticky="w")
        self.keybind_var = tk.StringVar(value=self.keybind)
        self.keybind_entry = ttk.Entry(panel, textvariable=self.keybind_var, width=14)
        self.keybind_entry.grid(row=1, column=3, sticky="w", padx=(8, 6))
        self.bind_btn = ttk.Button(panel, text="Satt tangent", command=self._set_keybind, style="App.TButton")
        self.bind_btn.grid(row=1, column=4, sticky="w")

        ttk.Label(panel, text="Mikrofon", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.input_var = tk.StringVar(value=self._input_label_for_index(self.input_device_index))
        self.input_combo = ttk.Combobox(panel, textvariable=self.input_var, values=[label for _, label in self.input_devices], state="readonly", width=56, style="App.TCombobox")
        self.input_combo.grid(row=2, column=1, columnspan=4, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.input_combo.bind("<<ComboboxSelected>>", self._on_input_changed)

        ttk.Label(panel, text="Hogtalare", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.output_var = tk.StringVar(value=self._output_label_for_index(self.output_device_index))
        self.output_combo = ttk.Combobox(panel, textvariable=self.output_var, values=[label for _, label in self.output_devices], state="readonly", width=56, style="App.TCombobox")
        self.output_combo.grid(row=3, column=1, columnspan=4, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.output_combo.bind("<<ComboboxSelected>>", self._on_output_changed)

        ttk.Label(panel, text="Svarslage", style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.response_mode_var = tk.StringVar(value="Tal + text" if self.tts_enabled else "Endast text")
        self.response_mode_combo = ttk.Combobox(
            panel,
            textvariable=self.response_mode_var,
            values=["Tal + text", "Endast text"],
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.response_mode_combo.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(10, 0))
        self.response_mode_combo.bind("<<ComboboxSelected>>", self._on_response_mode_changed)

        panel.columnconfigure(1, weight=1)

        controls = ttk.Frame(shell, style="App.TFrame", padding=(0, 12, 0, 12))
        controls.pack(fill=tk.X)
        self.start_btn = ttk.Button(controls, text="Start Listening", command=self.start, style="Primary.TButton")
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop, state=tk.DISABLED, style="Danger.TButton")
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(controls, text="Rensa text", command=self._clear_log, style="App.TButton").pack(side=tk.LEFT, padx=(10, 0))

        text_bar = ttk.Frame(shell, style="CardAlt.TFrame", padding=(12, 10))
        text_bar.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(text_bar, text="Skriv till assistenten:", style="Subtle.TLabel").pack(side=tk.LEFT)
        self.text_input_var = tk.StringVar()
        self.text_input_entry = ttk.Entry(text_bar, textvariable=self.text_input_var)
        self.text_input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.text_input_entry.bind("<Return>", self._send_text_query_event)
        self.send_btn = ttk.Button(text_bar, text="Skicka", command=self._send_text_query, style="App.TButton")
        self.send_btn.pack(side=tk.RIGHT)

        log_card = ttk.Frame(shell, style="Card.TFrame", padding=(0, 0, 0, 0))
        log_card.pack(fill=tk.BOTH, expand=True)
        title_row = ttk.Frame(log_card, style="Card.TFrame", padding=(14, 10, 14, 8))
        title_row.pack(fill=tk.X)
        ttk.Label(title_row, text="Activity", style="Section.TLabel").pack(side=tk.LEFT)

        self.log = scrolledtext.ScrolledText(
            log_card,
            wrap=tk.WORD,
            font=("Cascadia Code", 10),
            bg="#0f172a",
            fg="#dbeafe",
            insertbackground="#f8fafc",
            relief=tk.FLAT,
            borderwidth=0,
            padx=14,
            pady=10,
        )
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.tag_configure("system_prefix", foreground="#7dd3fc")
        self.log.tag_configure("user_prefix", foreground="#86efac")
        self.log.tag_configure("assistant_prefix", foreground="#f9a8d4")
        self.log.tag_configure("msg", foreground="#dbeafe")
        self.log.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_queue = queue.Queue()
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def _get_input_devices(self) -> list[tuple[int, str]]:
        devices = []
        try:
            for idx, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    devices.append((idx, d["name"]))
        except Exception:
            pass
        return devices

    def _get_output_devices(self) -> list[tuple[int, str]]:
        devices = []
        try:
            for idx, d in enumerate(sd.query_devices()):
                if d["max_output_channels"] > 0:
                    devices.append((idx, d["name"]))
        except Exception:
            pass
        return devices

    def _device_index_from_saved_name(self, saved_name: str, devices: list[tuple[int, str]]) -> int | None:
        if not saved_name:
            return None
        saved = normalize_text(saved_name)
        for idx, name in devices:
            if normalize_text(name) == saved:
                return idx
        for idx, name in devices:
            norm = normalize_text(name)
            if saved in norm or norm in saved:
                return idx
        return None

    def _pick_default_input_device(self) -> int | None:
        if not self.input_devices:
            return None
        bad_tokens = ["stereo mix", "capture wave", "stream", "vad"]
        good_tokens = ["microphone", "headset", "mic", "brio"]
        for idx, name in self.input_devices:
            low = name.lower()
            if any(b in low for b in bad_tokens):
                continue
            if any(g in low for g in good_tokens):
                return idx
        try:
            default_in = sd.default.device[0]
            if isinstance(default_in, int):
                return default_in
        except Exception:
            pass
        return self.input_devices[0][0]

    def _pick_default_output_device(self) -> int | None:
        if not self.output_devices:
            return None
        try:
            default_out = sd.default.device[1]
            if isinstance(default_out, int):
                return default_out
        except Exception:
            pass
        return self.output_devices[0][0]

    def _pick_input_device(self, saved_name: str) -> int | None:
        saved_idx = self._device_index_from_saved_name(saved_name, self.input_devices)
        if saved_idx is not None:
            return saved_idx
        return self._pick_default_input_device()

    def _pick_output_device(self, saved_name: str) -> int | None:
        saved_idx = self._device_index_from_saved_name(saved_name, self.output_devices)
        if saved_idx is not None:
            return saved_idx
        return self._pick_default_output_device()

    def _input_label_for_index(self, index: int | None) -> str:
        if index is None:
            return "Ingen mikrofon hittad"
        for idx, label in self.input_devices:
            if idx == index:
                return label
        return str(index)

    def _output_label_for_index(self, index: int | None) -> str:
        if index is None:
            return "Ingen hogtalare hittad"
        for idx, label in self.output_devices:
            if idx == index:
                return label
        return str(index)

    def _apply_audio_device_defaults(self) -> None:
        with self.lock:
            input_idx = self.input_device_index
            output_idx = self.output_device_index
        try:
            current = sd.default.device
            in_default = input_idx if input_idx is not None else current[0]
            out_default = output_idx if output_idx is not None else current[1]
            sd.default.device = (in_default, out_default)
        except Exception:
            pass

    def _on_input_changed(self, _event=None) -> None:
        selected = self.input_var.get()
        for idx, label in self.input_devices:
            if label == selected:
                with self.lock:
                    self.input_device_index = idx
                self._save_env("INPUT_DEVICE_NAME", label)
                self._apply_audio_device_defaults()
                self._log("system", f"Mikrofon satt till: {label}")
                return

    def _on_output_changed(self, _event=None) -> None:
        selected = self.output_var.get()
        for idx, label in self.output_devices:
            if label == selected:
                with self.lock:
                    self.output_device_index = idx
                self._save_env("OUTPUT_DEVICE_NAME", label)
                self._apply_audio_device_defaults()
                self._log("system", f"Hogtalare satt till: {label}")
                return

    def _save_env(self, key: str, value: str) -> None:
        if not self.env_path.exists():
            self.env_path.write_text("", encoding="utf-8")
        set_key(str(self.env_path), key, value)
        os.environ[key] = value

    def _on_response_mode_changed(self, _event=None) -> None:
        mode = self.response_mode_var.get().strip()
        enabled = mode == "Tal + text"
        with self.lock:
            self.tts_enabled = enabled
        self._save_env("TTS_ENABLED", "1" if enabled else "0")
        self._log("system", f"Svarslage satt till: {mode}")

    def _normalize_keybind_name(self, key: str) -> str:
        k = (key or "").strip().lower().replace(" ", "")
        aliases = {
            "xbutton1": "mouse4",
            "x1": "mouse4",
            "button.x1": "mouse4",
            "mousebutton4": "mouse4",
            "mb4": "mouse4",
            "xbutton2": "mouse5",
            "x2": "mouse5",
            "button.x2": "mouse5",
            "mousebutton5": "mouse5",
            "mb5": "mouse5",
            "left": "mouse1",
            "right": "mouse2",
            "middle": "mouse3",
        }
        return aliases.get(k, k)

    def _mouse_button_name(self, button) -> str:
        try:
            if button == pynput_mouse.Button.left:
                return "mouse1"
            if button == pynput_mouse.Button.right:
                return "mouse2"
            if button == pynput_mouse.Button.middle:
                return "mouse3"
            if button == pynput_mouse.Button.x1:
                return "mouse4"
            if button == pynput_mouse.Button.x2:
                return "mouse5"
        except Exception:
            pass
        return str(button).lower()

    def _on_keyboard_event(self, event) -> None:
        if not self.capturing_keybind:
            return
        try:
            if event.event_type == "down" and event.name:
                key = self._normalize_keybind_name(event.name)
                self.root.after(0, lambda: self._finish_keybind_capture(key, None))
        except Exception as e:
            self.root.after(0, lambda: self._finish_keybind_capture("", str(e)))

    def _on_mouse_click(self, _x, _y, button, pressed) -> None:
        name = self._mouse_button_name(button)
        with self.lock:
            if pressed:
                self.mouse_buttons_down.add(name)
            else:
                self.mouse_buttons_down.discard(name)

        if self.capturing_keybind and pressed:
            key = self._normalize_keybind_name(name)
            self.root.after(0, lambda: self._finish_keybind_capture(key, None))

    def _start_input_hooks(self) -> None:
        keyboard.hook(self._on_keyboard_event)
        self.mouse_listener = pynput_mouse.Listener(on_click=self._on_mouse_click)
        self.mouse_listener.daemon = True
        self.mouse_listener.start()

    def _is_bind_pressed(self, key_name: str) -> bool:
        normalized = self._normalize_keybind_name(key_name)
        if normalized.startswith("mouse"):
            with self.lock:
                return normalized in self.mouse_buttons_down
        try:
            return keyboard.is_pressed(normalized)
        except Exception:
            return False

    def _set_keybind(self) -> None:
        if self.capturing_keybind:
            return

        self.capturing_keybind = True
        self.bind_btn.configure(text="Tryck valfri tangent...")
        self._log("system", "Tryck nu den tangent du vill anvanda som push-to-talk.")

    def _finish_keybind_capture(self, key: str, error: str | None) -> None:
        self.capturing_keybind = False
        self.bind_btn.configure(text="Satt tangent")
        if error:
            self._log("system", f"Kunde inte lasa tangent: {error}")
            return
        if not key:
            self._log("system", "Ingen giltig tangent upptacktes.")
            return
        key = self._normalize_keybind_name(key)
        with self.lock:
            self.keybind = key
        self.keybind_var.set(key)
        self._save_env("PTT_KEYBIND", key)
        self._log("system", f"Push-to-talk satt till: {key}")

    def _on_profile_changed(self, _event=None) -> None:
        selected = self.profile_var.get()
        for profile in VOICE_PROFILES:
            if profile.label == selected:
                with self.lock:
                    self.current_profile = profile
                    self.tts_engine = self._make_local_tts_engine(profile.language)
                self._log("system", f"Profil vald: {profile.label}")
                break

    def _make_local_tts_engine(self, language: str):
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

    def start(self) -> None:
        if self.running:
            return
        self.stop_event.clear()
        self.running = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Status: listening")
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()
        self._log("system", "Lyssning startad.")

    def stop(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        self.running = False
        try:
            sd.stop()
        except Exception:
            pass
        try:
            with self.lock:
                self.tts_engine.stop()
        except Exception:
            pass
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Status: stopped")
        self._log("system", "Lyssning stoppad.")

    def _log(self, source: str, text: str) -> None:
        self.log_queue.put((source, text))

    def _queue_status(self, status_text: str) -> None:
        self.status_queue.put(status_text)

    def _drain_queues(self) -> None:
        while True:
            try:
                status_text = self.status_queue.get_nowait()
            except queue.Empty:
                break
            self.status_var.set(status_text)

        while True:
            try:
                source, text = self.log_queue.get_nowait()
            except queue.Empty:
                break
            prefix = {"user": "Du", "assistant": "Assistent", "system": "System"}.get(source, "Log")
            tag = {"user": "user_prefix", "assistant": "assistant_prefix", "system": "system_prefix"}.get(source, "system_prefix")
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, f"{prefix}: ", tag)
            self.log.insert(tk.END, f"{text}\n\n", "msg")
            self.log.see(tk.END)
            self.log.configure(state=tk.DISABLED)

        self.root.after(80, self._drain_queues)

    def _edge_voice_order(self, profile: VoiceProfile) -> list[str]:
        base = EDGE_FALLBACKS.get(profile.language, [])
        return list(dict.fromkeys([profile.primary_edge_voice] + base))

    async def _edge_save_audio(self, text: str, voice: str, out_path: str, rate: str, pitch: str) -> None:
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts saknas")
        communicate = Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        await communicate.save(out_path)

    def _speak(self, text: str, profile: VoiceProfile, engine) -> None:
        if self.stop_event.is_set():
            return
        chunks = split_for_tts(
            text,
            short_pauses=SHORT_PAUSES,
            reduce_comma_pauses=REDUCE_COMMA_PAUSES,
        )
        if not chunks:
            return

        if EDGE_TTS_AVAILABLE:
            for voice in self._edge_voice_order(profile):
                if self.stop_event.is_set():
                    return
                voice_ok = True
                for chunk in chunks:
                    if self.stop_event.is_set():
                        return
                    chunk_ok = False
                    last_error = None
                    for rate, pitch in EDGE_FALLBACK_PROSODY:
                        for _ in range(EDGE_RETRIES):
                            if self.stop_event.is_set():
                                return
                            tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name)
                            try:
                                asyncio.run(self._edge_save_audio(chunk, voice, str(tmp), rate, pitch))
                                audio, samplerate = sf.read(str(tmp), dtype="float32")
                                with self.lock:
                                    output_idx = self.output_device_index
                                sd.play(audio, samplerate, device=output_idx)
                                sd.wait()
                                chunk_ok = True
                                break
                            except Exception as e:
                                last_error = e
                                time.sleep(0.12)
                            finally:
                                tmp.unlink(missing_ok=True)
                        if chunk_ok:
                            break
                    if not chunk_ok:
                        voice_ok = False
                        self._log("system", f"Edge TTS misslyckades med {voice}. Provar nasta rost.")
                        break
                if voice_ok:
                    return

        self._log("system", "Edge TTS otillganglig/instabil just nu. Anvander lokal rost.")
        for chunk in chunks:
            if self.stop_event.is_set():
                return
            engine.say(chunk)
            engine.runAndWait()

    def _record_while_key_held(self, key_name: str) -> str:
        frames = []
        with self.lock:
            input_idx = self.input_device_index
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=BLOCK_SIZE,
                device=input_idx,
            ) as stream:
                while self.running and not self.stop_event.is_set() and self._is_bind_pressed(key_name):
                    chunk, overflowed = stream.read(BLOCK_SIZE)
                    if overflowed:
                        self._log("system", "Audio overflow upptackt.")
                    frames.append(chunk.copy())
                    time.sleep(0.001)
        except Exception as e:
            self._log("system", f"Kunde inte spela in ljud: {e}")
            return ""

        if not frames:
            return ""

        audio = np.concatenate(frames, axis=0)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        sf.write(tmp.name, audio, SAMPLE_RATE)
        return tmp.name

    def _transcribe(self, wav_path: str, language: str) -> str:
        with open(wav_path, "rb") as f:
            resp = self.client.audio.transcriptions.create(model="whisper-1", file=f, language=language)
        return (resp.text or "").strip()

    def _ask(self, user_text: str, language: str) -> str:
        system_prompt = system_prompt_for(language)
        for tool_type in ["web_search", "web_search_preview"]:
            try:
                resp = self.client.responses.create(
                    model="gpt-4.1-mini",
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    tools=[{"type": tool_type}],
                )
                out = (getattr(resp, "output_text", None) or "").strip()
                if out:
                    return out
            except Exception:
                continue
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def _run_loop(self) -> None:
        while self.running and not self.stop_event.is_set():
            with self.lock:
                profile = self.current_profile
                key_name = self.keybind
                engine = self.tts_engine

            try:
                if not self._is_bind_pressed(key_name):
                    time.sleep(0.02)
                    continue

                self._queue_status("Status: recording")
                self._log("system", f"Spelar in... (hall {key_name})")
                wav_path = self._record_while_key_held(key_name)

                if self.stop_event.is_set():
                    if wav_path:
                        Path(wav_path).unlink(missing_ok=True)
                    break

                self._queue_status("Status: processing")
                if not wav_path:
                    time.sleep(0.08)
                    continue

                try:
                    user_text = self._transcribe(wav_path, profile.language)
                finally:
                    Path(wav_path).unlink(missing_ok=True)

                if not user_text:
                    self._log("system", "Horde inget, forsok igen.")
                    self._queue_status("Status: listening")
                    continue

                self._process_user_text(user_text, profile, engine)
            except Exception as e:
                self._log("system", f"Fel i loop: {e}")
            finally:
                if self.running and not self.stop_event.is_set():
                    self._queue_status("Status: listening")
                time.sleep(0.05)

    def _process_user_text(self, user_text: str, profile: VoiceProfile, engine) -> None:
        with self.lock:
            tts_enabled = self.tts_enabled
        self._log("user", user_text)
        hue_reply = self.integrations.handle_hue_command(user_text)
        if hue_reply:
            self._log("assistant", hue_reply)
            if tts_enabled:
                self._queue_status("Status: speaking")
                self._speak(hue_reply, profile, engine)
            return
        spotify_reply = self.integrations.handle_spotify_command(user_text)
        if spotify_reply:
            self._log("assistant", spotify_reply)
            if tts_enabled:
                self._queue_status("Status: speaking")
                self._speak(spotify_reply, profile, engine)
            return
        vac_reply = self.integrations.handle_vacuum_command(user_text)
        if vac_reply:
            self._log("assistant", vac_reply)
            if tts_enabled:
                self._queue_status("Status: speaking")
                self._speak(vac_reply, profile, engine)
            return
        answer = self._ask(user_text, profile.language)
        self._log("assistant", answer)
        if tts_enabled:
            self._queue_status("Status: speaking")
            self._speak(answer, profile, engine)

    def _send_text_query_event(self, _event=None):
        self._send_text_query()

    def _send_text_query(self) -> None:
        text = self.text_input_var.get().strip()
        if not text:
            return
        self.text_input_var.set("")

        with self.lock:
            profile = self.current_profile
            engine = self.tts_engine

        def worker():
            try:
                self._queue_status("Status: processing")
                self._process_user_text(text, profile, engine)
            except Exception as e:
                self._log("system", f"Textfraga misslyckades: {e}")
            finally:
                if self.running and not self.stop_event.is_set():
                    self._queue_status("Status: listening")
                else:
                    self._queue_status("Status: idle")

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    ensure_tk_runtime_paths()
    root = tk.Tk()
    app = VoiceAssistantGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()

