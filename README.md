# Voice Assistant (Raspberry Pi 5)

This project is my own code for a local voice assistant that I run on my Raspberry Pi 5.

The assistant supports push-to-talk, speech-to-text, AI responses, text-to-speech, and smart-home/media control (Philips Hue, Spotify, and Home Assistant vacuum).

## Features

- Push-to-talk with configurable keyboard key or mouse button.
- Speech recognition using OpenAI (Whisper).
- AI responses using OpenAI.
- TTS responses (Edge TTS with local fallback).
- Philips Hue control:
  - on/off
  - brightness in percent
  - color changes (for example blue, red, indigo)
  - specific room/light targeting with name matching
- Spotify control:
  - play/pause
  - next/previous track
  - play a specific track by name
  - volume
  - seek to a specific timestamp in the current track
- Home Assistant vacuum control:
  - start, pause, return to dock, locate, status
- Selectable input device (microphone) and output device (speaker).
- Persists latest settings in `.env` (for example push-to-talk and audio devices).

## Project Structure

- `voice_assistant_gui.py`  
  Main GUI application with push-to-talk loop, recording, transcription, and command routing.

- `assistant_integrations.py`  
  Integrations for Hue, Spotify, and Home Assistant (vacuum).

- `assistant_text.py`  
  Text helpers: normalization, system prompts, TTS text cleanup, and chunking.

- `push_to_talk_openai_edge.py`  
  Older/alternative push-to-talk script.

- `.env`  
  Local environment variables and API secrets (should not be committed).

## Requirements

- Python 3.11+ (I run this in a virtual environment).
- Internet access for OpenAI/Spotify/Hue discovery.
- Raspberry Pi OS with desktop (for Tkinter GUI and global input hooks).
- API keys/tokens in `.env`.

Important `.env` values:

- `OPENAI_API_KEY`
- `HUE_BRIDGE_IP`, `HUE_APP_KEY`
- `HA_URL`, `HA_TOKEN`, `HA_VACUUM_ENTITY_ID`
- `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REFRESH_TOKEN`
- `SPOTIFY_DEVICE_ID` (optional)
- `PTT_KEYBIND`, `INPUT_DEVICE_NAME`, `OUTPUT_DEVICE_NAME`

## Run the App

Simple (Bash/Git Bash):

```bash
bash ./start
```

Simple (PowerShell):

```powershell
python .\voice_assistant_gui.py
```

If you use a virtual environment in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python .\voice_assistant_gui.py
```

## Raspberry Pi 5 Note

This project is built with Raspberry Pi 5 as the main target.  
Global keyboard/mouse hooks may require additional permissions depending on your Linux session (X11 vs Wayland).

## Security

- Never commit `.env` to Git.
- Rotate tokens/secrets if they are exposed.
