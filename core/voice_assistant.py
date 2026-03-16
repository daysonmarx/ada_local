"""
Voice Assistant - STT listener with wake word detection.
Captures speech and emits signals to the GUI, which routes
through ChatHandlers → Claude for processing and TTS.
"""

import re
from typing import Optional
from PySide6.QtCore import QObject, Signal

from config import GRAY, RESET, CYAN, GREEN, WAKE_WORD
from core.stt import STTListener
from core.tts import tts


class VoiceAssistant(QObject):
    """Voice assistant: listens for wake word, emits speech to GUI."""

    # Signals for UI
    wake_word_detected = Signal()
    speech_recognized = Signal(str)

    def __init__(self):
        super().__init__()
        self.stt_listener: Optional[STTListener] = None
        self.running = False

    def initialize(self) -> bool:
        """Initialize STT listener."""
        try:
            print(f"{CYAN}[VoiceAssistant] Initializing voice assistant components...{RESET}")
            self.stt_listener = STTListener(
                wake_word_callback=self._on_wake_word,
                speech_callback=self._on_speech
            )
            print(f"{CYAN}[VoiceAssistant] ✓ STT listener created{RESET}")

            if not self.stt_listener.initialize():
                print(f"{GRAY}[VoiceAssistant] ✗ Failed to initialize STT.{RESET}")
                return False
            print(f"{CYAN}[VoiceAssistant] ✓ STT initialized{RESET}")

            # Ensure TTS is initialized
            if not tts._initialized:
                print(f"{CYAN}[VoiceAssistant] Initializing TTS...{RESET}")
                tts.toggle(True)
                print(f"{CYAN}[VoiceAssistant] ✓ TTS initialized{RESET}")

            print(f"{CYAN}[VoiceAssistant] ✓ Voice assistant initialized successfully{RESET}")
            return True
        except Exception as e:
            print(f"{GRAY}[VoiceAssistant] ✗ Initialization error: {e}{RESET}")
            import traceback
            traceback.print_exc()
            return False

    def start(self):
        """Start the voice assistant."""
        if self.running:
            return

        if not self.stt_listener:
            if not self.initialize():
                return

        self.running = True
        self.stt_listener.start()
        print(f"{CYAN}[VoiceAssistant] Voice assistant started. Say '{GREEN}{WAKE_WORD}{RESET}{CYAN}' to activate.{RESET}")

    def stop(self):
        """Stop the voice assistant."""
        if not self.running:
            return

        self.running = False
        if self.stt_listener:
            self.stt_listener.stop()
        print(f"{GRAY}[VoiceAssistant] Voice assistant stopped.{RESET}")

    def _on_wake_word(self):
        """Handle wake word detection."""
        print(f"{GREEN}[VoiceAssistant] ✓ Wake word callback received!{RESET}")
        print(f"{GREEN}[VoiceAssistant] Emitting wake_word_detected signal...{RESET}")
        self.wake_word_detected.emit()
        print(f"{GREEN}[VoiceAssistant] ✓ Signal emitted. Listening for speech...{RESET}")

    def _on_speech(self, text: str):
        """Handle recognized speech after wake word.
        Emits speech_recognized signal — the GUI routes it through
        ChatHandlers.send_message() so it appears in the chat and
        uses the same LLM + TTS pipeline as typed messages.
        """
        if not text.strip():
            return

        # Remove wake word from text if present
        text = re.sub(rf'\b{WAKE_WORD}\b', '', text, flags=re.IGNORECASE).strip()
        text = text.lstrip(',').strip()
        if not text:
            return

        print(f"{CYAN}[VoiceAssistant] Speech: '{text}' — sending to chat UI{RESET}")
        self.speech_recognized.emit(text)


# Global voice assistant instance
voice_assistant = VoiceAssistant()
