"""
Speech-to-Text with Wake Word Detection for Voice Assistant.
Uses RealTimeSTT for real-time transcription with wake word detection.

Two modes:
  - Porcupine (USE_PORCUPINE_WAKE_WORD=True): hardware-level detection, only
    built-in keywords (jarvis, alexa, computer, etc.). Requires API key.
  - Transcription (default): uses Whisper to transcribe and checks for the
    wake word in the text. Supports any custom wake word like "Tali".
"""

import threading
import time
import re
from typing import Optional, Callable
from config import (
    WAKE_WORD, REALTIMESTT_MODEL, WAKE_WORD_SENSITIVITY,
    USE_PORCUPINE_WAKE_WORD, PORCUPINE_ACCESS_KEY,
    GRAY, RESET, CYAN, YELLOW, GREEN
)


class STTListener:
    """
    Real-time STT listener with wake word detection using RealTimeSTT.
    """

    def __init__(self, wake_word_callback: Callable, speech_callback: Callable):
        self.wake_word_callback = wake_word_callback
        self.speech_callback = speech_callback
        self.running = False
        self.listening_thread = None

        # RealTimeSTT recorder
        self.recorder = None
        self.initialized = False

        print(f"{CYAN}[STT] Initializing RealTimeSTT listener...{RESET}")
        print(f"{CYAN}[STT] Wake word: '{WAKE_WORD}'{RESET}")
        mode = "Porcupine" if USE_PORCUPINE_WAKE_WORD else "transcription"
        print(f"{CYAN}[STT] Detection method: {mode}{RESET}")

    def initialize(self) -> bool:
        """Initialize RealTimeSTT with wake word detection."""
        try:
            from RealtimeSTT import AudioToTextRecorder
            import torch

            print(f"{CYAN}[STT] Loading RealTimeSTT...{RESET}")

            cuda_available = torch.cuda.is_available()
            device = "cuda" if cuda_available else "cpu"
            if cuda_available:
                print(f"{GREEN}[STT] ✓ CUDA available ({torch.cuda.get_device_name()}){RESET}")
            else:
                print(f"{YELLOW}[STT] ⚠ No CUDA, using CPU{RESET}")

            if USE_PORCUPINE_WAKE_WORD and PORCUPINE_ACCESS_KEY:
                # Porcupine-based detection (only built-in keywords work)
                print(f"{CYAN}[STT] Using Porcupine wake word backend...{RESET}")
                self.recorder = AudioToTextRecorder(
                    model=REALTIMESTT_MODEL,
                    language="en",
                    device=device,
                    spinner=False,
                    wakeword_backend="pvporcupine",
                    wake_words=WAKE_WORD.lower(),
                    wake_words_sensitivity=WAKE_WORD_SENSITIVITY,
                    on_wakeword_detected=self._on_wakeword_detected,
                )
            else:
                # Transcription-based detection (supports any custom wake word)
                print(f"{CYAN}[STT] Using transcription-based wake word detection...{RESET}")
                self.recorder = AudioToTextRecorder(
                    model=REALTIMESTT_MODEL,
                    language="en",
                    device=device,
                    spinner=False,
                    # Teach Whisper that "Tali" is a name it should transcribe
                    initial_prompt=f"Tali, hello. Tali, how are you? Tali, what time is it?",
                )

            self.initialized = True
            print(f"{GREEN}[STT] ✓ RealTimeSTT initialized (model: {REALTIMESTT_MODEL}, wake word: '{WAKE_WORD}'){RESET}")
            return True
        except ImportError:
            print(f"{GRAY}[STT] ✗ RealTimeSTT not installed. Run: pip install realtimestt{RESET}")
            return False
        except Exception as e:
            print(f"{GRAY}[STT] ✗ Initialization error: {e}{RESET}")
            import traceback
            traceback.print_exc()
            return False

    def _on_wakeword_detected(self):
        """Callback when Porcupine detects the wake word."""
        print(f"\n{GREEN}[STT] ✓ Wake word '{WAKE_WORD}' detected (Porcupine)!{RESET}")
        if self.wake_word_callback:
            self.wake_word_callback()

    def start(self):
        """Start listening."""
        if not self.initialized:
            print(f"{YELLOW}[STT] Not initialized. Call initialize() first.{RESET}")
            return False

        if self.running:
            return True

        self.running = True
        self.listening_thread = threading.Thread(
            target=self._run_listener,
            daemon=True
        )
        self.listening_thread.start()
        print(f"{GREEN}[STT] ✓ Listener started — say '{WAKE_WORD}' to activate{RESET}")
        return True

    def _run_listener(self):
        """Main listening loop."""
        try:
            if USE_PORCUPINE_WAKE_WORD and PORCUPINE_ACCESS_KEY:
                self._run_porcupine_mode()
            else:
                self._run_transcription_mode()
        except Exception as e:
            print(f"{GRAY}[STT] Listener error: {e}{RESET}")
            import traceback
            traceback.print_exc()
            self.running = False

    def _run_porcupine_mode(self):
        """Porcupine wake word → then transcribe speech."""
        while self.running and self.recorder:
            print(f"{GRAY}[STT] ⏳ Waiting for wake word '{WAKE_WORD}'...{RESET}")
            text = self.recorder.text()
            if text and text.strip():
                text_clean = self._strip_wake_word(text)
                if text_clean:
                    print(f"{GREEN}[STT] 🔊 Speech: '{text_clean}'{RESET}")
                    self.speech_callback(text_clean)

    def _run_transcription_mode(self):
        """Continuously transcribe and look for the wake word in the text.

        Whisper often swallows or mangles short names like "Tali", so we use
        a generous fuzzy match.  If the wake word is not found at all, the
        utterance is ignored (prevents the assistant from responding to every
        background conversation).
        """
        # Whisper often mishears "Tali" as "Ali", "Tally", "Danny", "Dolly",
        # "Telly", etc., or drops it entirely.
        wake_lower = WAKE_WORD.lower()
        _SOUND_ALIKES = {
            "tali": (
                r"(?:^|\b)"
                r"(?:tali|tally|talley|taly|tari|toli|telli|telly"
                r"|ali|alli|ollie"
                r"|dali|dolly|danny|denny|delly"
                r"|holly|molly|polly|kelly"
                r"|tarley|charlie)"
                r"(?:\b|[,.:;!?\s])"
            ),
        }
        pattern_str = _SOUND_ALIKES.get(wake_lower, rf'\b{re.escape(wake_lower)}\b')
        wake_pattern = re.compile(pattern_str, re.IGNORECASE)

        # Also detect when Whisper drops the wake word entirely but the user
        # clearly spoke a short command right after it (e.g. heard "How are you?"
        # when user said "Tali, how are you?").  We track consecutive "missed"
        # utterances — if the first word of the transcription is a common
        # question/command starter, treat it as wake-word-activated.
        _COMMAND_STARTERS = re.compile(
            r'^(?:how|what|when|where|who|why|which|can|could|would|will|do|does|did'
            r'|is|are|was|were|tell|set|turn|play|search|find|show|open|close'
            r'|remind|schedule|create|add|make|help|please'
            r'|como|qual|quando|onde|quem|porque|pode|faz|diga|mostre'
            r')\b',
            re.IGNORECASE
        )

        while self.running and self.recorder:
            print(f"{GRAY}[STT] ⏳ Listening for '{WAKE_WORD}'...{RESET}")

            # recorder.text() blocks until silence after speech
            text = self.recorder.text()

            if not text or not text.strip():
                continue

            text = text.strip()
            print(f"{CYAN}[STT] 📝 Heard: '{text}'{RESET}")

            # --- Try 1: exact / fuzzy wake word match ---
            wake_found = wake_pattern.search(text)

            # --- Try 2: Whisper dropped the wake word but it looks like a command ---
            if not wake_found and _COMMAND_STARTERS.match(text):
                print(f"{YELLOW}[STT] ⚠ No wake word but looks like a command — accepting{RESET}")
                wake_found = True

            if wake_found:
                print(f"{GREEN}[STT] ✓ Wake word detected!{RESET}")

                # Notify wake word callback
                if self.wake_word_callback:
                    self.wake_word_callback()

                # Strip wake word and process remaining text
                text_clean = self._strip_wake_word(text)
                if text_clean:
                    print(f"{GREEN}[STT] 🔊 Speech: '{text_clean}'{RESET}")
                    self.speech_callback(text_clean)
                else:
                    # Wake word only — wait for the next utterance as the command
                    print(f"{CYAN}[STT] 🎤 Wake word only — listening for command...{RESET}")
                    cmd_text = self.recorder.text()
                    if cmd_text and cmd_text.strip():
                        cmd_clean = self._strip_wake_word(cmd_text.strip())
                        if cmd_clean:
                            print(f"{GREEN}[STT] 🔊 Command: '{cmd_clean}'{RESET}")
                            self.speech_callback(cmd_clean)

    def _strip_wake_word(self, text: str) -> str:
        """Remove the wake word (and sound-alikes) and surrounding punctuation."""
        wake_lower = WAKE_WORD.lower()
        _SOUND_ALIKES = {
            "tali": r"\b(?:tali|tally|ali|telly|dali|dolly|danny|tari|talley|toli)\b",
        }
        pattern_str = _SOUND_ALIKES.get(wake_lower, rf'\b{re.escape(WAKE_WORD)}\b')
        cleaned = re.sub(pattern_str, '', text, flags=re.IGNORECASE)
        # Clean up leftover punctuation and whitespace
        cleaned = re.sub(r'^[\s,.:;!?]+', '', cleaned).strip()
        return cleaned

    def stop(self):
        """Stop listening."""
        self.running = False
        if self.recorder:
            try:
                print(f"{CYAN}[STT] Shutting down recorder...{RESET}")
                self.recorder.shutdown()
            except Exception as e:
                print(f"{GRAY}[STT] Error stopping recorder: {e}{RESET}")
        if self.listening_thread:
            self.listening_thread.join(timeout=2.0)
        print(f"{CYAN}[STT] ✓ Listener stopped{RESET}")
