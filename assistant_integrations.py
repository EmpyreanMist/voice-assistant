import os
import re
import threading
import time
from typing import Callable

import requests
import urllib3

from assistant_text import match_tokens, normalize_text

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class AssistantIntegrations:
    def __init__(self, log_fn: Callable[[str, str], None], save_env_fn: Callable[[str, str], None]):
        self._log = log_fn
        self._save_env = save_env_fn

        self.hue_bridge_ip = os.getenv("HUE_BRIDGE_IP", "").strip()
        self.hue_app_key = os.getenv("HUE_APP_KEY", "").strip()

        self.ha_url = os.getenv("HA_URL", "").strip()
        self.ha_token = os.getenv("HA_TOKEN", "").strip()
        self.ha_vacuum_entity_id = os.getenv("HA_VACUUM_ENTITY_ID", "").strip()

        self.spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
        self.spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
        self.spotify_refresh_token = os.getenv("SPOTIFY_REFRESH_TOKEN", "").strip()
        self.spotify_device_id = os.getenv("SPOTIFY_DEVICE_ID", "").strip()
        self.spotify_market = os.getenv("SPOTIFY_MARKET", "SE").strip() or "SE"
        self.spotify_access_token = ""
        self.spotify_access_token_expires_at = 0.0

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
        self._ha_request("POST", f"/api/services/vacuum/{service}", {"entity_id": self.ha_vacuum_entity_id})

    def _ha_vacuum_state(self):
        if not self.ha_vacuum_entity_id:
            raise RuntimeError("HA_VACUUM_ENTITY_ID saknas.")
        return self._ha_request("GET", f"/api/states/{self.ha_vacuum_entity_id}")

    def _spotify_missing_fields(self) -> list[str]:
        missing = []
        if not self.spotify_client_id:
            missing.append("SPOTIFY_CLIENT_ID")
        if not self.spotify_client_secret:
            missing.append("SPOTIFY_CLIENT_SECRET")
        if not self.spotify_refresh_token:
            missing.append("SPOTIFY_REFRESH_TOKEN")
        return missing

    def _spotify_refresh_access_token(self) -> str:
        if self.spotify_access_token and time.time() < self.spotify_access_token_expires_at - 30:
            return self.spotify_access_token
        missing = self._spotify_missing_fields()
        if missing:
            raise RuntimeError(f"Spotify saknar konfiguration: {', '.join(missing)}.")

        payload = {"grant_type": "refresh_token", "refresh_token": self.spotify_refresh_token}
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data=payload,
            auth=(self.spotify_client_id, self.spotify_client_secret),
            timeout=10,
        )
        if resp.status_code == 400:
            raise RuntimeError("Spotify refresh token ogiltig eller utgangen.")
        resp.raise_for_status()
        data = resp.json() if resp.text else {}
        token = data.get("access_token")
        if not token:
            raise RuntimeError("Kunde inte hamta Spotify access token.")
        expires_in = int(data.get("expires_in", 3600))
        self.spotify_access_token = token
        self.spotify_access_token_expires_at = time.time() + max(60, expires_in)
        return token

    def _spotify_request(self, method: str, path: str, payload=None, params=None, retry_on_401: bool = True):
        token = self._spotify_refresh_access_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"https://api.spotify.com/v1{path}"
        resp = requests.request(method, url, json=payload, params=params, headers=headers, timeout=10)
        if resp.status_code == 401 and retry_on_401:
            self.spotify_access_token = ""
            self.spotify_access_token_expires_at = 0.0
            return self._spotify_request(method, path, payload=payload, params=params, retry_on_401=False)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "?")
            raise RuntimeError(f"Spotify rate limit, forsok igen om {retry_after} sekunder.")
        if resp.status_code >= 400:
            msg = ""
            try:
                body = resp.json()
                if isinstance(body, dict):
                    err = body.get("error")
                    if isinstance(err, dict):
                        msg = str(err.get("message", ""))
                    elif isinstance(err, str):
                        msg = err
            except Exception:
                msg = ""
            if resp.status_code == 403:
                raise RuntimeError("Spotify playback kraver Premium-konto.")
            raise RuntimeError(msg or f"Spotify API-fel ({resp.status_code}).")
        if resp.status_code == 204 or not resp.text:
            return {}
        return resp.json()

    def _spotify_get_devices(self) -> list[dict]:
        data = self._spotify_request("GET", "/me/player/devices")
        devices = (data or {}).get("devices", []) if isinstance(data, dict) else []
        if not isinstance(devices, list):
            return []
        return [d for d in devices if isinstance(d, dict)]

    def _spotify_resolve_device_id(self, avoid_device_id: str = "") -> str:
        avoid = (avoid_device_id or "").strip()
        configured = (self.spotify_device_id or "").strip()
        devices = self._spotify_get_devices()
        if not devices:
            return ""

        if configured:
            for d in devices:
                if str(d.get("id", "")).strip() == configured and configured != avoid:
                    return configured

        for d in devices:
            did = str(d.get("id", "")).strip()
            if did and did != avoid and bool(d.get("is_active")):
                return did

        for d in devices:
            did = str(d.get("id", "")).strip()
            if did and did != avoid:
                return did
        return ""

    def _spotify_transfer_playback(self, device_id: str, play: bool = False) -> None:
        if not device_id:
            return
        self._spotify_request("PUT", "/me/player", payload={"device_ids": [device_id], "play": bool(play)})

    def _spotify_request_with_device(self, method: str, path: str, payload=None, params=None):
        merged_params = dict(params or {})
        resolved_device_id = self._spotify_resolve_device_id()
        if resolved_device_id:
            merged_params["device_id"] = resolved_device_id
        try:
            return self._spotify_request(method, path, payload=payload, params=merged_params)
        except RuntimeError as e:
            msg = str(e).lower()
            if "device not found" not in msg:
                raise
            current_id = str(merged_params.get("device_id", "")).strip()
            if current_id:
                self._log("system", f"Spotify device hittades inte ({current_id}). Forsoker auto-valja enhet.")
            fallback_device_id = self._spotify_resolve_device_id(avoid_device_id=current_id)
            if not fallback_device_id:
                raise RuntimeError("Ingen tillganglig Spotify-enhet hittades. Oppna Spotify pa dator/mobil och forsok igen.")
            self._spotify_transfer_playback(fallback_device_id, play=False)
            merged_params["device_id"] = fallback_device_id
            return self._spotify_request(method, path, payload=payload, params=merged_params)

    def _spotify_search_track(self, query: str, artist: str = ""):
        if not query:
            return None
        queries = []
        track_part = query.strip()
        artist_part = artist.strip()
        if track_part and artist_part:
            queries.append(f'track:"{track_part}" artist:"{artist_part}"')
        queries.append(track_part)
        if track_part and artist_part:
            queries.append(f"{track_part} {artist_part}")

        query_track_norm = normalize_text(track_part)
        query_track_tokens = match_tokens(track_part)
        query_artist_norm = normalize_text(artist_part)
        query_artist_tokens = match_tokens(artist_part)
        best_any = None
        best_any_score = -1.0
        best_with_artist = None
        best_with_artist_score = -1.0

        def score_item(item: dict) -> tuple[float, float]:
            item_name = str(item.get("name", "")).strip()
            item_name_norm = normalize_text(item_name)
            item_track_tokens = match_tokens(item_name)
            artists_text = ", ".join(a.get("name", "") for a in item.get("artists", []) if a.get("name"))
            item_artist_norm = normalize_text(artists_text)
            item_artist_tokens = match_tokens(artists_text)

            track_overlap = 0.0
            if query_track_tokens:
                track_overlap = len(query_track_tokens & item_track_tokens) / max(1, len(query_track_tokens))
            artist_overlap = 0.0
            if query_artist_tokens:
                artist_overlap = len(query_artist_tokens & item_artist_tokens) / max(1, len(query_artist_tokens))

            score = 0.0
            score += track_overlap * 4.0
            if query_track_norm and query_track_norm == item_name_norm:
                score += 2.8
            elif query_track_norm and query_track_norm in item_name_norm:
                score += 1.4
            score += artist_overlap * 5.0
            if query_artist_norm and query_artist_norm in item_artist_norm:
                score += 2.5
            return score, artist_overlap

        for q in queries:
            res = self._spotify_request(
                "GET",
                "/search",
                params={"q": q, "type": "track", "limit": 10, "market": self.spotify_market},
            )
            tracks = (res or {}).get("tracks", {})
            items = tracks.get("items", []) if isinstance(tracks, dict) else []
            if not items:
                continue
            artist_low = normalize_text(artist_part)
            for item in items:
                score, artist_overlap = score_item(item)
                if score > best_any_score:
                    best_any = item
                    best_any_score = score
                if artist_low:
                    if artist_overlap > 0.0 and score > best_with_artist_score:
                        best_with_artist = item
                        best_with_artist_score = score

        if artist_low:
            return best_with_artist
        return best_any

    def _spotify_is_playing(self) -> bool | None:
        try:
            state = self._spotify_request("GET", "/me/player")
        except Exception:
            return None
        if not isinstance(state, dict):
            return None
        value = state.get("is_playing")
        if isinstance(value, bool):
            return value
        return None

    def _parse_time_to_ms(self, low_text: str) -> int | None:
        clock_match = re.search(r"\b(\d{1,2})\s*:\s*(\d{1,2})(?:\s*:\s*(\d{1,2}))?\b", low_text)
        if clock_match:
            a = int(clock_match.group(1))
            b = int(clock_match.group(2))
            c = int(clock_match.group(3)) if clock_match.group(3) else None
            total_seconds = a * 60 + b if c is None else a * 3600 + b * 60 + c
            return max(0, total_seconds * 1000)

        min_sec_match = re.search(
            r"\b(\d{1,3})\s*(?:min|minut|minuter|m)\b(?:\s*(?:och)?\s*(\d{1,2})\s*(?:sek|sekund|sekunder|s))?",
            low_text,
        )
        if min_sec_match:
            mins = int(min_sec_match.group(1))
            secs = int(min_sec_match.group(2) or 0)
            return max(0, (mins * 60 + secs) * 1000)

        sec_match = re.search(r"\b(\d{1,4})\s*(?:sek|sekund|sekunder|s)\b", low_text)
        if sec_match:
            secs = int(sec_match.group(1))
            return max(0, secs * 1000)
        return None

    def _extract_spotify_track_query(self, low_text: str) -> str:
        match = re.search(r"(?:spela|play|sla pa|starta)\s*[,:\-]?\s*(?:upp\s+)?(.+?)(?:\s+pa\s+spotify)?$", low_text)
        if not match:
            return ""
        query = match.group(1).strip()
        query = query.strip(",.:- ")
        query = re.sub(r"^(en|ett|the)\s+", "", query).strip()
        cleanup_suffixes = [r"\b(pa|i)\s+\d+\s*%", r"\bvid\b.*$", r"\bfran\b.*$", r"\btill\b.*$"]
        for pattern in cleanup_suffixes:
            query = re.sub(pattern, "", query).strip()
        query = re.sub(r"\s*,\s*", " ", query).strip()
        return query

    def _split_track_artist(self, query: str) -> tuple[str, str]:
        q = (query or "").strip()
        if not q:
            return "", ""
        m = re.search(r"^(.+?)\s+(?:med|by)\s+(.+)$", q)
        if not m:
            return q, ""
        return m.group(1).strip(), m.group(2).strip()

    def _looks_like_music_request(self, low: str) -> bool:
        if any(
            k in low for k in [
                "spotify", "pa spotify", "on spotify", "musik", "lat", "song", "playlist", "album", "artist"
            ]
        ):
            return True
        if any(k in low for k in ["nasta", "next", "foregaende", "previous", "pause", "pausa", "volym", "volume", "seek", "spola"]):
            return True
        # Typical spoken track request: "sla pa <track> med/by <artist>"
        if re.search(r"(?:spela|play|sla pa|starta)\s*[,:\-]?\s+.+\s+(?:med|by)\s+.+", low):
            return True
        return False

    def _hue_get_groups_lights(self):
        if not self.hue_bridge_ip or not self.hue_app_key:
            return {}, {}
        groups = self._hue_request("GET", f"/api/{self.hue_app_key}/groups")
        lights = self._hue_request("GET", f"/api/{self.hue_app_key}/lights")
        return groups, lights

    def _hue_find_target(self, text: str, groups: dict, lights: dict):
        low = normalize_text(text)
        if "alla" in low or "all" in low:
            return "group", "0", "alla lampor"
        if "lampor" in low or "lights" in low:
            return "group", "0", "alla lampor"

        user_tokens = match_tokens(text)
        best = None
        best_score = 0.0

        def score_name(name: str) -> float:
            name_norm = normalize_text(name)
            if not name_norm:
                return 0.0
            if name_norm in low:
                return 10.0
            name_tokens = match_tokens(name)
            if not name_tokens:
                return 0.0
            overlap = len(name_tokens & user_tokens)
            if overlap <= 0:
                return 0.0
            return overlap / max(1, len(name_tokens))

        for gid, g in groups.items():
            name = str(g.get("name", "")).strip()
            if not name:
                continue
            sc = score_name(name)
            if sc > best_score:
                best_score = sc
                best = ("group", gid, name)

        for lid, l in lights.items():
            name = str(l.get("name", "")).strip()
            if not name:
                continue
            sc = score_name(name)
            if sc > best_score:
                best_score = sc
                best = ("light", lid, name)

        if best and best_score >= 0.5:
            return best
        return None, None, None

    def _hue_color_payload(self, low: str) -> tuple[dict, str] | None:
        color_map = [
            (["indigo"], {"on": True, "hue": 50000, "sat": 220, "bri": 180}, "indigo"),
            (["magenta", "fuchsia"], {"on": True, "hue": 61000, "sat": 235, "bri": 205}, "magenta"),
            (["bla", "blå", "blue"], {"on": True, "hue": 46920, "sat": 254, "bri": 200}, "bla"),
            (["rod", "röd", "red"], {"on": True, "hue": 0, "sat": 254, "bri": 200}, "rod"),
            (["gron", "grön", "green"], {"on": True, "hue": 25500, "sat": 254, "bri": 200}, "gron"),
            (["gul", "yellow"], {"on": True, "hue": 12750, "sat": 254, "bri": 220}, "gul"),
            (["orange"], {"on": True, "hue": 8000, "sat": 254, "bri": 220}, "orange"),
            (["lila", "purple", "violett", "violet"], {"on": True, "hue": 56100, "sat": 230, "bri": 190}, "lila"),
            (["rosa", "pink"], {"on": True, "hue": 62000, "sat": 180, "bri": 210}, "rosa"),
            (["turkos", "cyan"], {"on": True, "hue": 33000, "sat": 230, "bri": 210}, "turkos"),
            (["vit", "white"], {"on": True, "sat": 0, "bri": 230}, "vit"),
            (["varmvit", "varm vit", "warm white"], {"on": True, "ct": 370, "sat": 0, "bri": 230}, "varmvit"),
            (["kallvit", "kall vit", "cold white", "cool white"], {"on": True, "ct": 220, "sat": 0, "bri": 230}, "kallvit"),
        ]
        for names, payload, label in color_map:
            if any(name in low for name in names):
                return dict(payload), label
        return None

    def auto_connect_hue(self) -> None:
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

    def auto_connect_vacuum(self) -> None:
        def worker():
            missing = self._ha_missing_fields()
            if missing:
                self._log("system", f"Home Assistant vacuum saknar konfiguration: {', '.join(missing)}")
                return
            try:
                st = self._ha_vacuum_state()
                attrs = st.get("attributes", {}) if isinstance(st, dict) else {}
                battery = attrs.get("battery_level", attrs.get("battery", "?"))
                state = st.get("state", "?") if isinstance(st, dict) else "?"
                self._log("system", f"Home Assistant vacuum ansluten: {self.ha_vacuum_entity_id} ({state}, {battery}% batteri).")
            except Exception as e:
                self._log("system", f"Home Assistant vacuum kunde inte verifieras: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def auto_connect_spotify(self) -> None:
        def worker():
            missing = self._spotify_missing_fields()
            if missing:
                self._log("system", f"Spotify saknar konfiguration: {', '.join(missing)}")
                return
            try:
                me = self._spotify_request("GET", "/me")
                display_name = (me or {}).get("display_name") or (me or {}).get("id") or "okand anvandare"
                self._log("system", f"Spotify ansluten: {display_name}.")
            except Exception as e:
                self._log("system", f"Spotify kunde inte verifieras: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def handle_vacuum_command(self, text: str) -> str | None:
        low = normalize_text(text)
        vac_entity_keywords = ["dammsugare", "robotdammsugare", "roborock", "vacuum"]
        if not any(k in low for k in vac_entity_keywords):
            return None

        missing = self._ha_missing_fields()
        if missing:
            return f"Home Assistant vacuum saknar konfiguration: {', '.join(missing)}."
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
            st = self._ha_vacuum_state()
            attrs = st.get("attributes", {}) if isinstance(st, dict) else {}
            battery = attrs.get("battery_level", attrs.get("battery", "?"))
            state = st.get("state", "?") if isinstance(st, dict) else "?"
            return f"Dammsugaren ar {state} och har {battery}% batteri."
        except Exception as e:
            return f"Dammsugar-kommando via Home Assistant misslyckades: {e}"

    def handle_spotify_command(self, text: str) -> str | None:
        low = normalize_text(text)
        spotify_keywords = [
            "spotify", "musik", "lat", "song", "playlist", "album", "artist",
            "spela", "play", "pausa", "pause", "nasta", "next", "foregaende", "previous",
            "volym", "volume", "skip", "hoppa", "vad spelas", "what is playing", "spola", "seek",
            "sla pa", "starta",
        ]
        if not any(k in low for k in spotify_keywords):
            return None
        has_music_context = self._looks_like_music_request(low)
        if not has_music_context and any(
            k in low for k in [
                "dammsugare", "dammsugaren", "roborock", "vacuum",
                "lampa", "lampor", "hue", "philips", "ljus"
            ]
        ):
            return None
        missing = self._spotify_missing_fields()
        if missing:
            return f"Spotify saknar konfiguration: {', '.join(missing)}."
        try:
            seek_intent = any(k in low for k in ["hoppa till", "spola till", "ga till", "go to", "seek", "fran borjan", "from start"])
            if seek_intent:
                position_ms = 0 if "fran borjan" in low or "from start" in low else self._parse_time_to_ms(low)
                if position_ms is None:
                    return "Jag hittade ingen tid. Säg t.ex. 'hoppa till 1:23' eller 'spola till 90 sekunder'."
                self._spotify_request_with_device("PUT", "/me/player/seek", params={"position_ms": int(position_ms)})
                total_seconds = int(round(position_ms / 1000))
                mm = total_seconds // 60
                ss = total_seconds % 60
                return f"Hoppade till {mm}:{ss:02d} i laten."

            volume_match = re.search(r"(?:volym|volume)\D{0,6}(\d{1,3})\s*%?", low)
            if volume_match:
                pct = max(0, min(100, int(volume_match.group(1))))
                self._spotify_request_with_device("PUT", "/me/player/volume", params={"volume_percent": pct})
                return f"Satte Spotify-volym till {pct}%."

            if any(k in low for k in ["pausa", "pause", "stoppa musiken", "stopp musiken"]):
                try:
                    self._spotify_request_with_device("PUT", "/me/player/pause")
                except Exception:
                    is_playing = self._spotify_is_playing()
                    if is_playing is None or is_playing:
                        raise
                return "Pausade Spotify."

            if any(k in low for k in ["nasta", "next", "hoppa over", "skip", "nasta lat"]):
                self._spotify_request_with_device("POST", "/me/player/next")
                return "Bytte till nasta lat pa Spotify."

            if any(k in low for k in ["foregaende", "previous", "forra laten"]):
                self._spotify_request_with_device("POST", "/me/player/previous")
                return "Bytte till forra laten pa Spotify."

            if "vad spelas" in low or "what is playing" in low or "nuvarande lat" in low:
                current = self._spotify_request("GET", "/me/player/currently-playing")
                item = (current or {}).get("item") if isinstance(current, dict) else None
                if not item:
                    return "Jag hittar ingen aktiv Spotify-uppspelning just nu."
                name = item.get("name", "okand lat")
                artists = ", ".join(a.get("name", "") for a in item.get("artists", []) if a.get("name")) or "okand artist"
                return f"Du spelar {name} med {artists}."

            query = self._extract_spotify_track_query(low)
            if query and query not in {"spotify", "musik"} and not any(k in low for k in ["spela spotify", "play spotify", "spela musik", "play music"]):
                track_name, artist_name = self._split_track_artist(query)
                track = self._spotify_search_track(track_name, artist=artist_name)
                if not track:
                    return f"Hittade ingen lat for '{query}' pa Spotify."
                uri = track.get("uri")
                if not uri:
                    return "Hittade en lat men fick ingen spelbar URI fran Spotify."
                self._spotify_request_with_device("PUT", "/me/player/play", payload={"uris": [uri]})
                name = track.get("name", "okand lat")
                artists = ", ".join(a.get("name", "") for a in track.get("artists", []) if a.get("name")) or "okand artist"
                return f"Spelar {name} med {artists} pa Spotify."

            if any(k in low for k in ["spela", "play", "sla pa", "starta", "ateruppta", "resume"]) and has_music_context:
                try:
                    self._spotify_request_with_device("PUT", "/me/player/play")
                except Exception:
                    is_playing = self._spotify_is_playing()
                    if is_playing is None or not is_playing:
                        raise
                return "Startade Spotify."
            return None
        except Exception as e:
            return f"Spotify-kommando misslyckades: {e}"

    def handle_hue_command(self, text: str) -> str | None:
        if not self.hue_bridge_ip or not self.hue_app_key:
            return None
        low = normalize_text(text)
        if self._looks_like_music_request(low) and "hue" not in low and "lampa" not in low and "lampor" not in low:
            return None
        if any(k in low for k in ["spotify", "musik", "lat", "song", "playlist", "album", "artist"]):
            return None
        color_match = self._hue_color_payload(low)
        hue_keywords = [
            "tand", "slack", "stang av", "sla pa", "dimma", "ljusstyrka",
            "turn on", "turn off", "brightness", "%", "farg", "färg", "color",
            "hue", "philips", "lampa", "lampor", "indigo", "bla", "blå", "blue",
            "rod", "röd", "red", "gron", "grön", "green", "gul", "orange",
            "lila", "violett", "purple", "rosa", "pink", "magenta", "fuchsia", "turkos", "cyan",
            "gor", "andra", "byt",
        ]
        if not any(k in low for k in hue_keywords) and not color_match:
            return None
        try:
            groups, lights = self._hue_get_groups_lights()
            target_type, target_id, target_name = self._hue_find_target(low, groups, lights)
            if not target_type or not target_id:
                known = []
                for _, g in groups.items():
                    n = str(g.get("name", "")).strip()
                    if n:
                        known.append(n)
                for _, l in lights.items():
                    n = str(l.get("name", "")).strip()
                    if n:
                        known.append(n)
                sample = ", ".join(known[:8]) if known else "inga namn hittades"
                return f"Jag hittade inte vilken Hue-lampa eller grupp du menar. Exempel hos dig: {sample}."

            pct_match = re.search(r"(\d{1,3})\s*%", low)
            payload = {}
            if pct_match:
                pct = max(1, min(100, int(pct_match.group(1))))
                payload["on"] = True
                payload["bri"] = max(1, min(254, round(pct * 254 / 100)))
                reply = f"Satte {target_name} till {pct}%."
            elif color_match:
                payload, color_label = color_match
                reply = f"Andrade {target_name} till {color_label}."
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
