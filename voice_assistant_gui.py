import asyncio
import os
import queue
import tempfile
import threading
import time
import math
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
TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "150"))


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
        self.root.configure(bg="#05070d")

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
        self.hue_connected = bool(self.integrations.hue_bridge_ip and self.integrations.hue_app_key)
        self.spotify_connected = bool(
            self.integrations.spotify_client_id
            and self.integrations.spotify_client_secret
            and self.integrations.spotify_refresh_token
        )
        self.service_badges: dict[str, dict[str, object]] = {}
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
        self.mic_level_queue: queue.Queue[float] = queue.Queue()
        self.ai_level_queue: queue.Queue[float] = queue.Queue()
        self.mic_level_current = 0.0
        self.ai_level_current = 0.0
        self.lock = threading.Lock()

        self._start_input_hooks()
        self._apply_audio_device_defaults()
        self._setup_style()
        self._build_ui()
        if self.integrations.hue_bridge_ip and self.integrations.hue_app_key:
            self._log("system", f"Hue ansluten ({self.integrations.hue_bridge_ip}).")
        else:
            self._log("system", "Hue inte ansluten ännu. Försöker ansluta automatiskt.")
        if self.integrations.ha_enabled and self.integrations.ha_url and self.integrations.ha_token and self.integrations.ha_vacuum_entity_id:
            self._log("system", f"Home Assistant vacuum konfigurerad ({self.integrations.ha_vacuum_entity_id}).")
        if self.integrations.spotify_client_id and self.integrations.spotify_client_secret and self.integrations.spotify_refresh_token:
            self._log("system", "Spotify konfigurerad.")
        self._log("system", "Redo. Tryck Start och håll vald push-to-talk-tangent.")
        self.integrations.auto_connect_hue()
        self.integrations.auto_connect_vacuum()
        self.integrations.auto_connect_spotify()
        self.root.after(80, self._drain_queues)

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#060b16")
        style.configure("Hero.TFrame", background="#060b16")
        style.configure("Card.TFrame", background="#0e1628")
        style.configure("CardAlt.TFrame", background="#111e35")

        style.configure("App.TLabel", background="#060b16", foreground="#d8e3f8", font=("Bahnschrift", 10))
        style.configure("Card.TLabel", background="#0e1628", foreground="#d8e3f8", font=("Bahnschrift", 10))
        style.configure("Title.TLabel", background="#060b16", foreground="#f4f7ff", font=("Bahnschrift SemiBold", 30))
        style.configure("Subtle.TLabel", background="#060b16", foreground="#93a7cc", font=("Bahnschrift", 10))
        style.configure("Section.TLabel", background="#0e1628", foreground="#a9c0ef", font=("Bahnschrift SemiBold", 10))
        style.configure("Status.TLabel", background="#111e35", foreground="#d7ecff", font=("Bahnschrift SemiBold", 10))
        style.configure("StatusHint.TLabel", background="#060b16", foreground="#7d93bc", font=("Bahnschrift", 10))
        style.configure("Meter.TLabel", background="#111e35", foreground="#9ab4e5", font=("Bahnschrift", 9))

        style.configure("App.TButton", font=("Bahnschrift SemiBold", 10), padding=(13, 10), borderwidth=0)
        style.map("App.TButton", background=[("active", "#2a426f"), ("!disabled", "#1c2d4a")], foreground=[("!disabled", "#e6f0ff")])

        style.configure("Primary.TButton", font=("Bahnschrift SemiBold", 10), padding=(16, 11), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#18a669"), ("!disabled", "#22c07a")], foreground=[("!disabled", "#041b0f")])

        style.configure("Danger.TButton", font=("Bahnschrift SemiBold", 10), padding=(13, 10), borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#b0293a"), ("!disabled", "#d74255")], foreground=[("!disabled", "#fff4f6")])

        style.configure("App.TCombobox", padding=8, fieldbackground="#111e35", background="#111e35", foreground="#ecf3ff")
        style.map("App.TCombobox", fieldbackground=[("readonly", "#111e35")], selectbackground=[("readonly", "#1f3155")], selectforeground=[("readonly", "#ecf3ff")])
        style.configure("App.TEntry", fieldbackground="#111e35", background="#111e35", foreground="#ecf3ff", insertcolor="#f0f5ff")

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame", padding=18)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell, style="Hero.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        header_left = ttk.Frame(header, style="Hero.TFrame")
        header_left.grid(row=0, column=0, sticky="ew", padx=(0, 14))
        self.header_title = ttk.Label(header_left, text="Voice Assistant", style="Title.TLabel")
        self.header_title.pack(anchor="w")
        self.status_var = tk.StringVar(value="Status: idle")
        status_pill = ttk.Frame(header, style="CardAlt.TFrame", padding=(12, 7))
        status_pill.grid(row=0, column=1, sticky="ne", pady=(4, 0))
        self.status_dot = tk.Canvas(status_pill, width=10, height=10, bg="#111e35", bd=0, highlightthickness=0)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 7))
        self.status_dot_id = self.status_dot.create_oval(1, 1, 9, 9, fill="#6b7fa6", outline="")
        ttk.Label(status_pill, textvariable=self.status_var, style="Status.TLabel").pack()
        self.status_hint_var = tk.StringVar(value="Tryck Start och håll in push-to-talk för att prata.")
        self.header_subtitle = ttk.Label(
            header_left,
            text="Push-to-talk med OpenAI, Edge TTS, Hue, Spotify och Home Assistant",
            style="Subtle.TLabel",
            justify=tk.LEFT,
        )
        self.header_subtitle.pack(anchor="w", pady=(0, 2))
        self.header_hint = ttk.Label(
            header_left,
            textvariable=self.status_hint_var,
            style="StatusHint.TLabel",
            justify=tk.LEFT,
        )
        self.header_hint.pack(anchor="w")
        header_left.bind("<Configure>", self._on_header_resize)

        content = ttk.Frame(shell, style="App.TFrame")
        content.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left = ttk.Frame(content, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        settings_card = ttk.Frame(left, style="Card.TFrame", padding=14)
        settings_card.pack(fill=tk.X)
        ttk.Label(settings_card, text="Inställningar", style="Section.TLabel").grid(row=0, column=0, sticky="w", columnspan=3, pady=(0, 10))

        ttk.Label(settings_card, text="Röstprofil", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        self.profile_var = tk.StringVar(value=self.current_profile.label)
        self.profile_combo = ttk.Combobox(
            settings_card,
            textvariable=self.profile_var,
            values=[p.label for p in VOICE_PROFILES],
            state="readonly",
            width=30,
            style="App.TCombobox",
        )
        self.profile_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0))
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_changed)

        ttk.Label(settings_card, text="Push-to-talk", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.keybind_var = tk.StringVar(value=self.keybind)
        self.keybind_entry = ttk.Entry(settings_card, textvariable=self.keybind_var, width=10, style="App.TEntry")
        self.keybind_entry.grid(row=2, column=1, sticky="w", padx=(8, 6), pady=(10, 0))
        self.bind_btn = ttk.Button(settings_card, text="Byt", command=self._set_keybind, style="App.TButton")
        self.bind_btn.grid(row=2, column=2, sticky="w", pady=(10, 0))

        ttk.Label(settings_card, text="Mikrofon", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.input_var = tk.StringVar(value=self._input_label_for_index(self.input_device_index))
        self.input_combo = ttk.Combobox(
            settings_card,
            textvariable=self.input_var,
            values=[label for _, label in self.input_devices],
            state="readonly",
            width=30,
            style="App.TCombobox",
        )
        self.input_combo.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.input_combo.bind("<<ComboboxSelected>>", self._on_input_changed)

        ttk.Label(settings_card, text="Högtalare", style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.output_var = tk.StringVar(value=self._output_label_for_index(self.output_device_index))
        self.output_combo = ttk.Combobox(
            settings_card,
            textvariable=self.output_var,
            values=[label for _, label in self.output_devices],
            state="readonly",
            width=30,
            style="App.TCombobox",
        )
        self.output_combo.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.output_combo.bind("<<ComboboxSelected>>", self._on_output_changed)

        ttk.Label(settings_card, text="Svarsläge", style="Card.TLabel").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.response_mode_var = tk.StringVar(value="Tal + text" if self.tts_enabled else "Endast text")
        self.response_mode_combo = ttk.Combobox(
            settings_card,
            textvariable=self.response_mode_var,
            values=["Tal + text", "Endast text"],
            state="readonly",
            width=30,
            style="App.TCombobox",
        )
        self.response_mode_combo.grid(row=5, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.response_mode_combo.bind("<<ComboboxSelected>>", self._on_response_mode_changed)

        settings_card.columnconfigure(1, weight=1)

        controls = ttk.Frame(left, style="App.TFrame", padding=(0, 12, 0, 10))
        controls.pack(fill=tk.X)
        self.start_btn = ttk.Button(controls, text="Starta", command=self.start, style="Primary.TButton")
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(controls, text="Stoppa", command=self.stop, state=tk.DISABLED, style="Danger.TButton")
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(controls, text="Rensa text", command=self._clear_log, style="App.TButton").pack(side=tk.LEFT, padx=(10, 0))

        tips_card = ttk.Frame(left, style="CardAlt.TFrame", padding=(14, 12))
        tips_card.pack(fill=tk.X)
        ttk.Label(tips_card, text="Snabbtips", style="Section.TLabel").pack(anchor="w")
        ttk.Label(tips_card, text="1. Tryck Start", style="Card.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(tips_card, text="2. Håll in vald push-to-talk", style="Card.TLabel").pack(anchor="w")
        ttk.Label(tips_card, text="3. Släpp och få svar direkt", style="Card.TLabel").pack(anchor="w")
        ttk.Separator(tips_card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 8))
        ttk.Label(tips_card, text="Anslutna tjänster", style="Section.TLabel").pack(anchor="w")
        services_wrap = tk.Frame(tips_card, bg="#111e35", bd=0, highlightthickness=0)
        services_wrap.pack(fill=tk.X, pady=(4, 0))
        self._build_service_badge(services_wrap, "hue", "Hue Bridge")
        self._build_service_badge(services_wrap, "spotify", "Spotify")
        self._refresh_service_badges()

        right = ttk.Frame(content, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        meter_card = ttk.Frame(right, style="CardAlt.TFrame", padding=(12, 10))
        meter_card.grid(row=0, column=0, sticky="ew")
        meter_head = ttk.Frame(meter_card, style="CardAlt.TFrame")
        meter_head.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(meter_head, text="Ljudnivå", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Label(meter_head, text="Mic", style="Meter.TLabel").pack(side=tk.RIGHT, padx=(12, 0))
        ttk.Label(meter_head, text="AI", style="Meter.TLabel").pack(side=tk.RIGHT)
        self.level_canvas = tk.Canvas(meter_card, height=18, bg="#0b1222", highlightthickness=1, highlightbackground="#1d2b46", bd=0)
        self.level_canvas.pack(fill=tk.X)
        self.level_canvas.bind("<Configure>", lambda _e: self._draw_shared_level())

        text_bar = ttk.Frame(right, style="CardAlt.TFrame", padding=(12, 10))
        text_bar.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        ttk.Label(text_bar, text="Skriv till assistenten", style="Subtle.TLabel").pack(side=tk.LEFT)
        self.text_input_var = tk.StringVar()
        self.text_input_entry = ttk.Entry(text_bar, textvariable=self.text_input_var, style="App.TEntry")
        self.text_input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.text_input_entry.bind("<Return>", self._send_text_query_event)
        self.send_btn = ttk.Button(text_bar, text="Skicka", command=self._send_text_query, style="App.TButton")
        self.send_btn.pack(side=tk.RIGHT)

        log_card = ttk.Frame(right, style="Card.TFrame", padding=(0, 0, 0, 0))
        log_card.grid(row=2, column=0, sticky="nsew")
        title_row = ttk.Frame(log_card, style="Card.TFrame", padding=(14, 10, 14, 8))
        title_row.pack(fill=tk.X)
        ttk.Label(title_row, text="Konversation", style="Section.TLabel").pack(side=tk.LEFT)

        self.log = scrolledtext.ScrolledText(
            log_card,
            wrap=tk.WORD,
            font=("Cascadia Code", 10),
            bg="#0b1222",
            fg="#d7def5",
            insertbackground="#f3f6ff",
            relief=tk.FLAT,
            borderwidth=0,
            padx=14,
            pady=10,
        )
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.tag_configure("system_prefix", foreground="#62c8ff")
        self.log.tag_configure("user_prefix", foreground="#8af5bb")
        self.log.tag_configure("assistant_prefix", foreground="#ffd286")
        self.log.tag_configure("system_msg", foreground="#d7e9ff")
        self.log.tag_configure("user_msg", foreground="#ddfce9")
        self.log.tag_configure("assistant_msg", foreground="#fff4db")
        self.log.configure(state=tk.DISABLED)
        self._apply_status_visuals("Status: idle")

    def _on_header_resize(self, event) -> None:
        wrap = max(260, int(event.width) - 8)
        if hasattr(self, "header_subtitle"):
            self.header_subtitle.configure(wraplength=wrap)
        if hasattr(self, "header_hint"):
            self.header_hint.configure(wraplength=wrap)

    def _build_service_badge(self, parent, service_key: str, title: str) -> None:
        row = tk.Frame(parent, bg="#0d1a30", bd=0, highlightthickness=1, highlightbackground="#213a62")
        row.pack(fill=tk.X, pady=(6, 0))
        icon = tk.Canvas(row, width=24, height=24, bg="#0d1a30", bd=0, highlightthickness=0)
        icon.pack(side=tk.LEFT, padx=(8, 8), pady=7)
        self._draw_service_logo(icon, service_key)
        tk.Label(row, text=title, fg="#dbe9ff", bg="#0d1a30", font=("Bahnschrift SemiBold", 10)).pack(side=tk.LEFT)
        status_var = tk.StringVar(value="Ej ansluten")
        status_label = tk.Label(row, textvariable=status_var, fg="#7b90b8", bg="#0d1a30", font=("Bahnschrift", 9))
        status_label.pack(side=tk.RIGHT, padx=(8, 8))
        self.service_badges[service_key] = {"status_var": status_var, "status_label": status_label}

    def _draw_service_logo(self, canvas: tk.Canvas, service_key: str) -> None:
        if service_key == "spotify":
            canvas.create_oval(1, 1, 23, 23, fill="#1db954", outline="")
            canvas.create_arc(5, 7, 20, 16, start=20, extent=140, style=tk.ARC, outline="#0a1611", width=2)
            canvas.create_arc(6, 10, 19, 18, start=20, extent=135, style=tk.ARC, outline="#0a1611", width=2)
            canvas.create_arc(7, 13, 17, 19, start=20, extent=130, style=tk.ARC, outline="#0a1611", width=2)
            return
        canvas.create_oval(2, 2, 12, 12, fill="#f74f78", outline="")
        canvas.create_oval(12, 2, 22, 12, fill="#ffd054", outline="")
        canvas.create_oval(2, 12, 12, 22, fill="#6ee3ff", outline="")
        canvas.create_oval(12, 12, 22, 22, fill="#84ff9d", outline="")

    def _set_service_connected(self, service_key: str, connected: bool) -> None:
        if service_key == "hue":
            self.hue_connected = bool(connected)
        elif service_key == "spotify":
            self.spotify_connected = bool(connected)
        badge = self.service_badges.get(service_key)
        if not badge:
            return
        status_var = badge["status_var"]
        status_label = badge["status_label"]
        status_var.set("Ansluten" if connected else "Ej ansluten")
        status_label.configure(fg="#8af5bb" if connected else "#7b90b8")

    def _refresh_service_badges(self) -> None:
        self._set_service_connected("hue", self.hue_connected)
        self._set_service_connected("spotify", self.spotify_connected)

    def _update_service_state_from_log(self, source: str, text: str) -> None:
        if source != "system":
            return
        low = normalize_text(text)
        if "hue ansluten" in low or "hue konfigurerad" in low:
            self._set_service_connected("hue", True)
        elif "hue inte ansluten" in low or "hue-fel" in low:
            self._set_service_connected("hue", False)

        if "spotify ansluten" in low or "spotify konfigurerad" in low:
            self._set_service_connected("spotify", True)
        elif "spotify saknar konfiguration" in low or "spotify kunde inte verifieras" in low:
            self._set_service_connected("spotify", False)

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
                self._log("system", f"Högtalare satt till: {label}")
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
        self._log("system", f"Svarsläge satt till: {mode}")

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
        self._log("system", "Tryck nu den tangent du vill använda som push-to-talk.")

    def _finish_keybind_capture(self, key: str, error: str | None) -> None:
        self.capturing_keybind = False
        self.bind_btn.configure(text="Satt tangent")
        if error:
            self._log("system", f"Kunde inte läsa tangent: {error}")
            return
        if not key:
            self._log("system", "Ingen giltig tangent upptäcktes.")
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
        self._apply_status_visuals("Status: listening")
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
        self._apply_status_visuals("Status: stopped")
        self._queue_mic_level(0.0)
        self._queue_ai_level(0.0)
        self._log("system", "Lyssning stoppad.")

    def _log(self, source: str, text: str) -> None:
        self.log_queue.put((source, text))

    def _queue_status(self, status_text: str) -> None:
        self.status_queue.put(status_text)

    def _queue_mic_level(self, level: float) -> None:
        clamped = max(0.0, min(1.0, float(level)))
        try:
            while self.mic_level_queue.qsize() > 3:
                self.mic_level_queue.get_nowait()
        except Exception:
            pass
        self.mic_level_queue.put(clamped)

    def _queue_ai_level(self, level: float) -> None:
        clamped = max(0.0, min(1.0, float(level)))
        try:
            while self.ai_level_queue.qsize() > 3:
                self.ai_level_queue.get_nowait()
        except Exception:
            pass
        self.ai_level_queue.put(clamped)

    def _normalize_rms_level(self, rms: float, floor_db: float = -56.0, ceil_db: float = -16.0) -> float:
        if rms <= 1e-7:
            return 0.0
        db = 20.0 * math.log10(rms)
        norm = (db - floor_db) / (ceil_db - floor_db)
        norm = max(0.0, min(1.0, norm))
        if norm < 0.08:
            return 0.0
        return norm ** 1.35

    def _apply_status_visuals(self, status_text: str) -> None:
        raw = (status_text or "").strip().lower()
        state = raw.replace("status:", "").strip()
        labels = {
            "idle": ("Redo", "Tryck Start och håll in push-to-talk för att prata.", "#6b7fa6"),
            "listening": ("Lyssnar", "Väntar på att du trycker push-to-talk.", "#4de28b"),
            "recording": ("Spelar in", "Prata nu. Släpp push-to-talk när du är klar.", "#2ed3f1"),
            "processing": ("Tänker", "Transkriberar och förbereder svar.", "#f2c94c"),
            "speaking": ("Pratar", "Svarar med vald röstprofil.", "#ff9f67"),
            "stopped": ("Stoppad", "Lyssning är stoppad tills du startar igen.", "#ff6b88"),
        }
        label, hint, color = labels.get(state, ("Aktiv", "Assistent kör.", "#7fa0ff"))
        self.status_var.set(f"Status: {label}")
        self.status_hint_var.set(hint)
        if hasattr(self, "status_dot") and hasattr(self, "status_dot_id"):
            self.status_dot.itemconfigure(self.status_dot_id, fill=color)

    def _draw_shared_level(self) -> None:
        if not hasattr(self, "level_canvas"):
            return
        c = self.level_canvas
        c.delete("all")
        w = max(10, c.winfo_width())
        h = max(10, c.winfo_height())
        c.create_rectangle(0, 0, w, h, fill="#0b1222", outline="")
        is_mic = self.mic_level_current >= self.ai_level_current
        level = self.mic_level_current if is_mic else self.ai_level_current
        fill = int((w - 6) * max(0.0, min(1.0, level)))
        if fill > 0:
            color = "#18d7ff" if is_mic else "#ffac5f"
            c.create_rectangle(3, 3, 3 + fill, h - 3, fill=color, outline="")

    def _set_mic_level(self, level: float) -> None:
        target = max(0.0, min(1.0, float(level)))
        attack = 0.62
        release = 0.38
        alpha = attack if target > self.mic_level_current else release
        self.mic_level_current = self.mic_level_current + (target - self.mic_level_current) * alpha
        if self.mic_level_current < 0.02:
            self.mic_level_current = 0.0
        self._draw_shared_level()

    def _set_ai_level(self, level: float) -> None:
        target = max(0.0, min(1.0, float(level)))
        attack = 0.60
        release = 0.36
        alpha = attack if target > self.ai_level_current else release
        self.ai_level_current = self.ai_level_current + (target - self.ai_level_current) * alpha
        if self.ai_level_current < 0.02:
            self.ai_level_current = 0.0
        self._draw_shared_level()

    def _drain_queues(self) -> None:
        while True:
            try:
                status_text = self.status_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_status_visuals(status_text)

        mic_updated = False
        while True:
            try:
                level = self.mic_level_queue.get_nowait()
            except queue.Empty:
                break
            self._set_mic_level(level)
            mic_updated = True
        if not mic_updated:
            self._set_mic_level(self.mic_level_current * 0.45)

        ai_updated = False
        while True:
            try:
                level = self.ai_level_queue.get_nowait()
            except queue.Empty:
                break
            self._set_ai_level(level)
            ai_updated = True
        if not ai_updated:
            self._set_ai_level(self.ai_level_current * 0.45)

        while True:
            try:
                source, text = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._update_service_state_from_log(source, text)
            prefix = {"user": "Du", "assistant": "Assistent", "system": "System"}.get(source, "Log")
            tag = {"user": "user_prefix", "assistant": "assistant_prefix", "system": "system_prefix"}.get(source, "system_prefix")
            msg_tag = {"user": "user_msg", "assistant": "assistant_msg", "system": "system_msg"}.get(source, "system_msg")
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, f"{prefix}: ", tag)
            self.log.insert(tk.END, f"{text}\n\n", msg_tag)
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

    def _start_ai_meter_from_audio(self, audio: np.ndarray, samplerate: int) -> threading.Event:
        stop_flag = threading.Event()
        audio_arr = np.asarray(audio, dtype=np.float32)
        if audio_arr.ndim > 1:
            audio_arr = np.mean(audio_arr, axis=1)
        if samplerate <= 0 or audio_arr.size == 0:
            self._queue_ai_level(0.0)
            return stop_flag

        window = max(1, int(samplerate * 0.050))
        hop = max(1, int(samplerate * 0.035))

        def worker():
            i = 0
            n = int(audio_arr.shape[0])
            while i < n and not stop_flag.is_set() and not self.stop_event.is_set():
                seg = audio_arr[i:min(n, i + window)]
                if seg.size == 0:
                    level = 0.0
                else:
                    rms = float(np.sqrt(np.mean(np.square(seg))))
                    level = self._normalize_rms_level(rms, floor_db=-60.0, ceil_db=-18.0)
                self._queue_ai_level(level)
                time.sleep(hop / float(samplerate))
                i += hop
            self._queue_ai_level(0.0)

        threading.Thread(target=worker, daemon=True).start()
        return stop_flag

    def _speak(self, text: str, profile: VoiceProfile, engine) -> None:
        if self.stop_event.is_set():
            return
        chunks = split_for_tts(
            text,
            short_pauses=SHORT_PAUSES,
            reduce_comma_pauses=REDUCE_COMMA_PAUSES,
            max_chars=max(80, min(260, TTS_MAX_CHARS)),
            language=profile.language,
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
                                audio_arr = np.asarray(audio, dtype=np.float32)
                                meter_stop = self._start_ai_meter_from_audio(audio_arr, int(samplerate))
                                with self.lock:
                                    output_idx = self.output_device_index
                                sd.play(audio, samplerate, device=output_idx)
                                sd.wait()
                                meter_stop.set()
                                self._queue_ai_level(0.0)
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
                        self._log("system", f"Edge TTS misslyckades med {voice}. Provar nästa röst.")
                        break
                if voice_ok:
                    return

        self._log("system", "Edge TTS otillgänglig/instabil just nu. Använder lokal röst.")
        for chunk in chunks:
            if self.stop_event.is_set():
                return
            self._queue_ai_level(0.22)
            engine.say(chunk)
            engine.runAndWait()
            self._queue_ai_level(0.0)
        self._queue_ai_level(0.0)

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
                        self._log("system", "Audio overflow upptäckt.")
                    mono = chunk.astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(np.square(mono))))
                    level = self._normalize_rms_level(rms, floor_db=-56.0, ceil_db=-16.0)
                    self._queue_mic_level(level)
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
                self._log("system", f"Spelar in... (håll {key_name})")
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
                    self._log("system", "Hörde inget, försök igen.")
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
        spotify_reply = self.integrations.handle_spotify_command(user_text)
        if spotify_reply:
            self._log("assistant", spotify_reply)
            if tts_enabled:
                self._queue_status("Status: speaking")
                self._speak(spotify_reply, profile, engine)
            return
        hue_reply = self.integrations.handle_hue_command(user_text)
        if hue_reply:
            self._log("assistant", hue_reply)
            if tts_enabled:
                self._queue_status("Status: speaking")
                self._speak(hue_reply, profile, engine)
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
                self._log("system", f"Textfråga misslyckades: {e}")
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

