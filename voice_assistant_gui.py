import asyncio
import ipaddress
import os
import queue
import re
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext

import keyboard
import numpy as np
import pyttsx3
import requests
import sounddevice as sd
import soundfile as sf
from pynput import mouse as pynput_mouse
from dotenv import load_dotenv, set_key
from openai import OpenAI
import urllib3
from tkinter import simpledialog

try:
    from edge_tts import Communicate
    EDGE_TTS_AVAILABLE = True
except Exception:
    Communicate = None
    EDGE_TTS_AVAILABLE = False

try:
    from miio import RoborockVacuum
    ROBOROCK_AVAILABLE = True
except Exception:
    RoborockVacuum = None
    ROBOROCK_AVAILABLE = False

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
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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


def _clean_text_for_tts(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{2,}", ". ", cleaned)
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("**", "").replace("`", "").replace("#", "")
    cleaned = cleaned.replace("* ", "").replace("- ", "")
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("\\[", "").replace("\\]", "")
    cleaned = re.sub(r"\$([^$]+)\$", r"\1", cleaned)
    cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
    if SHORT_PAUSES:
        # Ta bort starka skiljetecken for kortare pauser.
        cleaned = cleaned.replace(";", " ")
        cleaned = cleaned.replace(":", " ")
        cleaned = re.sub(r"[.!?]\s+", " ", cleaned)
    if REDUCE_COMMA_PAUSES:
        # Vissa roster gor lang paus efter komma; mildra det.
        cleaned = re.sub(r",\s*", " ", cleaned)
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


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def _valid_host(value: str) -> bool:
    value = (value or "").strip()
    if not value:
        return False
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        pass
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}[A-Za-z0-9]", value))


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
        self.keybind = "enter"
        self.hue_bridge_ip = os.getenv("HUE_BRIDGE_IP", "").strip()
        self.hue_app_key = os.getenv("HUE_APP_KEY", "").strip()
        self.vacuum_ip = os.getenv("VACUUM_IP", "").strip()
        self.vacuum_token = os.getenv("VACUUM_TOKEN", "").strip()
        self.ha_url = os.getenv("HA_URL", "").strip()
        self.ha_token = os.getenv("HA_TOKEN", "").strip()
        self.ha_vacuum_entity_id = os.getenv("HA_VACUUM_ENTITY_ID", "").strip()
        self.vacuum = None
        self.input_devices = self._get_input_devices()
        self.input_device_index = self._pick_default_input_device()
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
        self._setup_style()
        self._build_ui()
        if self.hue_bridge_ip and self.hue_app_key:
            self._log("system", f"Hue ansluten ({self.hue_bridge_ip}).")
        else:
            self._log("system", "Hue inte ansluten ännu. Försöker ansluta automatiskt.")
        if self.ha_url and self.ha_token and self.ha_vacuum_entity_id:
            self._log("system", f"Home Assistant vacuum konfigurerad ({self.ha_vacuum_entity_id}).")
        self._log("system", "Redo. Tryck Start och hall vald push-to-talk-tangent.")
        self._auto_connect_hue()
        self._auto_connect_vacuum()
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
        ttk.Label(shell, text="Push-to-talk med OpenAI, Edge TTS och Hue-kontroll", style="Subtle.TLabel").pack(anchor="w", pady=(2, 14))

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

        ttk.Label(panel, text="Svarslage", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.response_mode_var = tk.StringVar(value="Tal + text" if self.tts_enabled else "Endast text")
        self.response_mode_combo = ttk.Combobox(
            panel,
            textvariable=self.response_mode_var,
            values=["Tal + text", "Endast text"],
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.response_mode_combo.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(10, 0))
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

    def _input_label_for_index(self, index: int | None) -> str:
        if index is None:
            return "Ingen mikrofon hittad"
        for idx, label in self.input_devices:
            if idx == index:
                return label
        return str(index)

    def _on_input_changed(self, _event=None) -> None:
        selected = self.input_var.get()
        for idx, label in self.input_devices:
            if label == selected:
                with self.lock:
                    self.input_device_index = idx
                self._log("system", f"Mikrofon satt till: {label}")
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

    def _discover_hue_bridge_ip(self) -> str:
        resp = requests.get("https://discovery.meethue.com", timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise RuntimeError("Ingen Hue Bridge hittades pa natverket.")
        return data[0]["internalipaddress"]

    def _hue_request(self, method: str, path: str, payload=None):
        if not self.hue_bridge_ip:
            raise RuntimeError("Hue bridge-IP saknas.")
        url = f"https://{self.hue_bridge_ip}{path}"
        resp = requests.request(method, url, json=payload, timeout=10, verify=False)
        resp.raise_for_status()
        if not resp.text:
            return {}
        return resp.json()

    def _ha_is_configured(self) -> bool:
        return bool(self.ha_url and self.ha_token and self.ha_vacuum_entity_id)

    def _ha_mode_requested(self) -> bool:
        return bool(self.ha_url or self.ha_token or self.ha_vacuum_entity_id)

    def _ha_missing_fields(self) -> list[str]:
        missing = []
        if not self.ha_url:
            missing.append("HA_URL")
        if not self.ha_token:
            missing.append("HA_TOKEN")
        if not self.ha_vacuum_entity_id:
            missing.append("HA_VACUUM_ENTITY_ID")
        return missing

    def _ha_request(self, method: str, path: str, payload=None):
        if not self.ha_url or not self.ha_token:
            raise RuntimeError("HA_URL/HA_TOKEN saknas.")
        base_url = self.ha_url.rstrip("/")
        headers = {"Authorization": f"Bearer {self.ha_token}", "Content-Type": "application/json"}
        resp = requests.request(method, f"{base_url}{path}", json=payload, headers=headers, timeout=10)
        if resp.status_code == 401:
            raise RuntimeError("Home Assistant token ogiltig eller utgangen.")
        resp.raise_for_status()
        if not resp.text:
            return {}
        return resp.json()

    def _ha_call_vacuum_service(self, service: str):
        if not self.ha_vacuum_entity_id:
            raise RuntimeError("HA_VACUUM_ENTITY_ID saknas.")
        self._ha_request(
            "POST",
            f"/api/services/vacuum/{service}",
            {"entity_id": self.ha_vacuum_entity_id},
        )

    def _ha_vacuum_state(self):
        if not self.ha_vacuum_entity_id:
            raise RuntimeError("HA_VACUUM_ENTITY_ID saknas.")
        return self._ha_request("GET", f"/api/states/{self.ha_vacuum_entity_id}")

    def _connect_hue(self) -> None:
        def worker():
            try:
                self._log("system", "Soker Hue Bridge...")
                if not self.hue_bridge_ip:
                    self.hue_bridge_ip = self._discover_hue_bridge_ip()
                    self._save_env("HUE_BRIDGE_IP", self.hue_bridge_ip)
                    self._log("system", f"Hittade Hue Bridge: {self.hue_bridge_ip}")

                if self.hue_app_key:
                    self._log("system", "Hue app-nyckel finns redan.")
                    return

                self._log("system", "Tryck pa knappen pa Hue-hubben nu (inom 30 sekunder)...")
                result = self._hue_request("POST", "/api", {"devicetype": "voice_assistant_gui#pc"})
                if not isinstance(result, list) or not result:
                    raise RuntimeError("Ovantat svar fran Hue.")
                first = result[0]
                if "error" in first:
                    desc = first["error"].get("description", "okant fel")
                    if "link button not pressed" in desc.lower():
                        raise RuntimeError("Tryck pa knappen pa Hue-hubben och forsok igen.")
                    raise RuntimeError(f"Pairing misslyckades: {desc}")
                username = first.get("success", {}).get("username")
                if not username:
                    raise RuntimeError("Kunde inte hamta app-nyckel.")
                self.hue_app_key = username
                self._save_env("HUE_APP_KEY", username)
                self._log("system", "Hue ansluten klart.")
            except Exception as e:
                self._log("system", f"Hue-fel: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _auto_connect_hue(self) -> None:
        self._connect_hue()

    def _connect_vacuum(self) -> None:
        if self._ha_mode_requested():
            missing = self._ha_missing_fields()
            if missing:
                self._log("system", f"Home Assistant-lage aktivt men saknar: {', '.join(missing)}")
                return
            self._log("system", "Home Assistant används för dammsugaren. Uppdatera HA_* i .env vid behov.")
            return
        if not ROBOROCK_AVAILABLE:
            self._log("system", "Roborock-stod saknas (python-miio). Installera beroenden.")
            return

        ip = simpledialog.askstring("Koppla Robo", "Dammsugarens IP-adress:", initialvalue=self.vacuum_ip or "")
        if not ip:
            return
        token = simpledialog.askstring("Koppla Robo", "Token (32 tecken):", initialvalue=self.vacuum_token or "")
        if not token:
            return

        ip = ip.strip()
        token = token.strip()
        if not _valid_host(ip):
            self._log("system", "Ogiltig VACUUM_IP. Ange lokal IP, t.ex. 192.168.1.45.")
            return
        if not re.fullmatch(r"[0-9a-fA-F]{32}", token):
            self._log("system", "Ogiltig VACUUM_TOKEN. Den ska vara 32 hex-tecken.")
            return
        self._save_env("VACUUM_IP", ip)
        self._save_env("VACUUM_TOKEN", token)
        self.vacuum_ip = ip
        self.vacuum_token = token

        def worker():
            try:
                vac = RoborockVacuum(ip, token)
                status = vac.status()
                with self.lock:
                    self.vacuum = vac
                self._log("system", f"Roborock ansluten ({ip}). Batteri: {getattr(status, 'battery', '?')}%")
            except Exception as e:
                self._log("system", f"Kunde inte ansluta Roborock: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _auto_connect_vacuum(self) -> None:
        def worker():
            if self._ha_mode_requested():
                missing = self._ha_missing_fields()
                if missing:
                    self._log("system", f"Home Assistant-lage aktivt men saknar: {', '.join(missing)}")
                    return
                try:
                    st = self._ha_vacuum_state()
                    attrs = st.get("attributes", {}) if isinstance(st, dict) else {}
                    battery = attrs.get("battery_level", attrs.get("battery", "?"))
                    state = st.get("state", "?") if isinstance(st, dict) else "?"
                    self._log("system", f"Home Assistant vacuum ansluten: {self.ha_vacuum_entity_id} ({state}, {battery}% batteri).")
                except Exception as e:
                    self._log("system", f"Home Assistant vacuum kunde inte verifieras: {e}")
                return

            if not ROBOROCK_AVAILABLE:
                return
            if not self.vacuum_ip or not self.vacuum_token:
                return
            if not _valid_host(self.vacuum_ip) or not re.fullmatch(r"[0-9a-fA-F]{32}", self.vacuum_token):
                return
            try:
                vac = RoborockVacuum(self.vacuum_ip, self.vacuum_token)
                status = vac.status()
                with self.lock:
                    self.vacuum = vac
                self._log("system", f"Roborock auto-ansluten ({self.vacuum_ip}). Batteri: {getattr(status, 'battery', '?')}%")
            except Exception as e:
                self._log("system", f"Roborock auto-anslutning misslyckades: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _get_vacuum(self):
        if not ROBOROCK_AVAILABLE:
            raise RuntimeError("python-miio/Roborock-stod saknas.")
        with self.lock:
            if self.vacuum is not None:
                return self.vacuum
        if not self.vacuum_ip or not self.vacuum_token:
            raise RuntimeError("VACUUM_IP/VACUUM_TOKEN saknas i .env.")
        if not _valid_host(self.vacuum_ip):
            raise RuntimeError("Ogiltig VACUUM_IP. Använd dammsugarens lokala IP-adress.")
        if not re.fullmatch(r"[0-9a-fA-F]{32}", self.vacuum_token):
            raise RuntimeError("Ogiltig VACUUM_TOKEN. Förväntar 32 hex-tecken.")
        vac = RoborockVacuum(self.vacuum_ip, self.vacuum_token)
        with self.lock:
            self.vacuum = vac
        return vac

    def _handle_vacuum_command(self, text: str) -> str | None:
        low = _normalize_text(text)
        vac_keywords = [
            "dammsugare", "roborock", "vacuum", "stada", "stada", "stadar",
            "starta", "borja", "pausa", "stopp", "docka", "hem", "hitta"
        ]
        if not any(k in low for k in vac_keywords):
            return None

        if self._ha_mode_requested():
            missing = self._ha_missing_fields()
            if missing:
                return f"Home Assistant-lage aktivt men saknar: {', '.join(missing)}."
            try:
                if any(k in low for k in ["starta", "borja", "stada", "clean", "resume"]):
                    self._ha_call_vacuum_service("start")
                    return "Startade dammsugaren via Home Assistant."
                if any(k in low for k in ["pausa", "pause", "stopp"]):
                    self._ha_call_vacuum_service("pause")
                    return "Pausade dammsugaren via Home Assistant."
                if any(k in low for k in ["docka", "hem", "ladda", "charge", "return"]):
                    self._ha_call_vacuum_service("return_to_base")
                    return "Skickade dammsugaren till dockan via Home Assistant."
                if any(k in low for k in ["hitta", "find", "where are you"]):
                    self._ha_call_vacuum_service("locate")
                    return "Bad dammsugaren spela upp ett ljud via Home Assistant."
                if any(k in low for k in ["status", "batteri", "battery"]):
                    st = self._ha_vacuum_state()
                    attrs = st.get("attributes", {}) if isinstance(st, dict) else {}
                    battery = attrs.get("battery_level", attrs.get("battery", "?"))
                    state = st.get("state", "?") if isinstance(st, dict) else "?"
                    return f"Dammsugaren ar {state} och har {battery}% batteri."
                st = self._ha_vacuum_state()
                attrs = st.get("attributes", {}) if isinstance(st, dict) else {}
                battery = attrs.get("battery_level", attrs.get("battery", "?"))
                state = st.get("state", "?") if isinstance(st, dict) else "?"
                return f"Dammsugaren ar {state} och har {battery}% batteri."
            except Exception as e:
                return f"Dammsugar-kommando via Home Assistant misslyckades: {e}"

        try:
            vac = self._get_vacuum()
            if any(k in low for k in ["starta", "borja", "stada", "clean", "resume"]):
                vac.start()
                return "Startade dammsugaren."
            if any(k in low for k in ["pausa", "pause", "stopp"]):
                vac.pause()
                return "Pausade dammsugaren."
            if any(k in low for k in ["docka", "hem", "ladda", "charge", "return"]):
                vac.home()
                return "Skickade dammsugaren till dockan."
            if any(k in low for k in ["hitta", "find", "where are you"]):
                vac.find()
                return "Bad dammsugaren spela upp ett ljud."
            if any(k in low for k in ["status", "batteri", "battery"]):
                st = vac.status()
                return f"Dammsugaren har {getattr(st, 'battery', '?')}% batteri."
            st = vac.status()
            return f"Dammsugaren har {getattr(st, 'battery', '?')}% batteri."
        except Exception as e:
            return f"Dammsugar-kommando misslyckades: {e}"

    def _hue_get_groups_lights(self):
        if not self.hue_bridge_ip or not self.hue_app_key:
            return {}, {}
        groups = self._hue_request("GET", f"/api/{self.hue_app_key}/groups")
        lights = self._hue_request("GET", f"/api/{self.hue_app_key}/lights")
        return groups, lights

    def _hue_find_target(self, text: str, groups: dict, lights: dict):
        low = _normalize_text(text)
        if "alla" in low or "all" in low:
            return "group", "0", "alla lampor"
        for gid, g in groups.items():
            name = str(g.get("name", "")).strip()
            if name and _normalize_text(name) in low:
                return "group", gid, name
        for lid, l in lights.items():
            name = str(l.get("name", "")).strip()
            if name and _normalize_text(name) in low:
                return "light", lid, name
        return "group", "0", "alla lampor"

    def _handle_hue_command(self, text: str) -> str | None:
        if not self.hue_bridge_ip or not self.hue_app_key:
            return None
        low = _normalize_text(text)
        hue_keywords = [
            "tand", "slack", "stang av", "sla pa", "dimma", "ljusstyrka",
            "turn on", "turn off", "brightness", "%"
        ]
        if not any(k in low for k in hue_keywords):
            return None

        try:
            groups, lights = self._hue_get_groups_lights()
            target_type, target_id, target_name = self._hue_find_target(low, groups, lights)

            pct_match = re.search(r"(\d{1,3})\s*%", low)
            payload = {}
            if pct_match:
                pct = max(1, min(100, int(pct_match.group(1))))
                payload["on"] = True
                payload["bri"] = max(1, min(254, round(pct * 254 / 100)))
                reply = f"Satte {target_name} till {pct}%."
            elif any(k in low for k in ["slack", "stang av", "turn off"]):
                payload["on"] = False
                reply = f"Slackte {target_name}."
            elif any(k in low for k in ["tand", "sla pa", "turn on"]):
                payload["on"] = True
                reply = f"Tande {target_name}."
            else:
                return None

            if target_type == "group":
                self._hue_request("PUT", f"/api/{self.hue_app_key}/groups/{target_id}/action", payload)
            else:
                self._hue_request("PUT", f"/api/{self.hue_app_key}/lights/{target_id}/state", payload)
            return reply
        except Exception as e:
            return f"Hue-kommando misslyckades: {e}"

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
        chunks = _split_for_tts(text)
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
                                sd.play(audio, samplerate)
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
        hue_reply = self._handle_hue_command(user_text)
        if hue_reply:
            self._log("assistant", hue_reply)
            if tts_enabled:
                self._queue_status("Status: speaking")
                self._speak(hue_reply, profile, engine)
            return
        vac_reply = self._handle_vacuum_command(user_text)
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

