"""
TTS (Text-to-Speech) module with pluggable providers.
Supports Kokoro (free, local, fast) and Edge TTS (free, online fallback).
Auto-detects language (English/Portuguese) and selects the appropriate voice.

Strategy: Collect the full LLM response, then synthesize and play it as one
continuous audio clip.  This avoids choppy playback caused by many small
synthesis calls with gaps between them.
"""

import asyncio
import io
import re
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import (
    TTS_PROVIDER, TTS_DEFAULT_LANG,
    KOKORO_VOICES,
    EDGE_TTS_VOICES,
)

# ANSI colors for console output
GRAY = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# Kokoro output sample rate (fixed by the model)
_KOKORO_SR = 24000

# Common Portuguese words for language detection
_PT_WORDS = {
    "olá", "oi", "obrigado", "obrigada", "bom", "boa", "dia", "noite", "tarde",
    "como", "você", "voce", "está", "tudo", "bem", "sim", "não", "nao", "por",
    "favor", "para", "isso", "aqui", "agora", "ainda", "mais", "muito", "também",
    "porque", "quando", "onde", "qual", "quem", "fazer", "pode", "preciso",
    "eu", "ele", "ela", "nós", "eles", "elas", "meu", "minha", "seu", "sua",
    "com", "sem", "mas", "que", "uma", "dos", "das", "nos", "nas", "aos",
    "pela", "pelo", "são", "tem", "há", "foi", "ser", "ter", "estar",
    "então", "entao", "já", "sempre", "nunca", "hoje", "amanhã", "ontem",
    "certo", "claro", "verdade", "belo", "bonito", "bonita", "feliz",
}

_PT_CHARS = set("àáâãçéêíóôõúü")


def detect_language(text: str) -> str:
    """Detect whether text is Portuguese or English. Returns 'pt' or 'en'."""
    text_lower = text.lower()
    # Check for Portuguese-specific accented characters
    if any(c in _PT_CHARS for c in text_lower):
        return "pt"
    # Check word overlap — require stronger signal to avoid false positives
    words = set(re.findall(r'\b\w+\b', text_lower))
    # Exclude very short/ambiguous words and proper names that overlap
    _AMBIGUOUS = {"para", "com", "sem", "que", "uma", "tem", "mas", "tali"}
    pt_matches = (words & _PT_WORDS) - _AMBIGUOUS
    # Need at least 2 strong Portuguese words, or 1 if text is very short (<=2 words)
    if len(pt_matches) >= 2 or (len(pt_matches) == 1 and len(words) <= 2):
        return "pt"
    return "en"


class SentenceBuffer:
    """Buffers streaming text and extracts complete sentences."""

    SENTENCE_ENDINGS = re.compile(r'([.!?])\s+|([.!?])$')

    def __init__(self):
        self.buffer = ""

    def add(self, text):
        self.buffer += text
        sentences = []
        while True:
            match = self.SENTENCE_ENDINGS.search(self.buffer)
            if match:
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()
                if sentence:
                    sentences.append(sentence)
                self.buffer = self.buffer[end_pos:]
            else:
                break
        return sentences

    def flush(self):
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None


# ---------------------------------------------------------------------------
# Provider: Kokoro TTS (free, local, fast)
# ---------------------------------------------------------------------------

class _KokoroProvider:
    """Synthesize speech using Kokoro TTS (hexgrad/Kokoro-82M).
    Runs entirely locally — no API key needed, very fast on Apple Silicon.
    """

    def __init__(self):
        from kokoro import KPipeline
        print(f"{CYAN}[TTS] Loading Kokoro English pipeline...{RESET}")
        self._en_pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        print(f"{CYAN}[TTS] Loading Kokoro Portuguese pipeline...{RESET}")
        self._pt_pipe = KPipeline(lang_code="p", repo_id="hexgrad/Kokoro-82M")

    def synthesize(self, text: str, interrupt_event: threading.Event):
        """Return (audio_data, samplerate) or None."""
        lang = detect_language(text)
        voice = KOKORO_VOICES.get(lang, KOKORO_VOICES.get(TTS_DEFAULT_LANG))
        pipeline = self._pt_pipe if lang == "pt" else self._en_pipe

        # Kokoro yields chunks — collect them all into one array
        audio_chunks = []
        for _gs, _ps, audio in pipeline(text, voice=voice):
            if interrupt_event.is_set():
                return None
            audio_chunks.append(audio)

        if interrupt_event.is_set() or not audio_chunks:
            return None

        audio_data = np.concatenate(audio_chunks)
        return audio_data, _KOKORO_SR


# ---------------------------------------------------------------------------
# Provider: Edge TTS (free, online — fallback)
# ---------------------------------------------------------------------------

class _EdgeTTSProvider:
    """Synthesize speech using Microsoft Edge TTS (free, no API key)."""

    def __init__(self):
        import edge_tts  # noqa: F401 – ensure available
        self._edge_tts = edge_tts

    def synthesize(self, text: str, interrupt_event: threading.Event):
        """Return (audio_data, samplerate) or None."""
        lang = detect_language(text)
        voice = EDGE_TTS_VOICES.get(lang, EDGE_TTS_VOICES.get(TTS_DEFAULT_LANG))

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._synthesize_async(text, voice, interrupt_event))
        finally:
            loop.close()

    async def _synthesize_async(self, text, voice, interrupt_event):
        communicate = self._edge_tts.Communicate(text, voice)
        audio_bytes = io.BytesIO()
        async for chunk in communicate.stream():
            if interrupt_event.is_set():
                return None
            if chunk["type"] == "audio":
                audio_bytes.write(chunk["data"])
        if interrupt_event.is_set() or audio_bytes.getbuffer().nbytes == 0:
            return None
        audio_bytes.seek(0)
        audio_data, samplerate = sf.read(audio_bytes)
        return audio_data, samplerate


# ---------------------------------------------------------------------------
# Main TTS class
# ---------------------------------------------------------------------------

# How long to wait (seconds) after the last queued sentence before we consider
# the response "complete" and send the collected text to the TTS provider.
_COLLECT_TIMEOUT = 8.0

# Sentinel value: when queued, tells the worker to speak everything collected so far
_FLUSH_SENTINEL = "__FLUSH__"


class SmartTTS:
    """TTS engine with pluggable provider and language auto-detection.

    Instead of synthesizing each sentence individually (which causes choppy
    playback due to per-call latency), this class collects all sentences
    from a streaming LLM response and synthesizes them as one continuous block.
    """

    def __init__(self):
        self.enabled = False
        self.speech_queue = queue.Queue()  # receives individual sentences
        self.worker_thread = None
        self.running = False
        self.interrupt_event = threading.Event()
        self._playback_done = threading.Event()  # signaled when playback finishes
        self._playback_done.set()  # initially "done" (nothing playing)
        self.current_playback = False
        self.available = True
        self._initialized = False
        self._provider = None

    def initialize(self):
        """Initialize the TTS provider."""
        try:
            provider_name = TTS_PROVIDER.lower()

            if provider_name == "kokoro":
                print(f"{CYAN}[TTS] Initializing Kokoro TTS (local, hexgrad/Kokoro-82M)...{RESET}")
                self._provider = _KokoroProvider()
                label = "Kokoro (local)"
            elif provider_name == "edge":
                print(f"{CYAN}[TTS] Initializing Edge TTS...{RESET}")
                self._provider = _EdgeTTSProvider()
                label = "Edge TTS"
            else:
                print(f"{YELLOW}[TTS] Unknown provider '{provider_name}', falling back to Kokoro{RESET}")
                self._provider = _KokoroProvider()
                label = "Kokoro (local)"

            self.running = True
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()
            self._initialized = True

            print(f"{GREEN}[TTS] ✓ {label} ready (auto-detect: en / pt-BR){RESET}")
            return True

        except Exception as e:
            print(f"{YELLOW}[TTS] Failed to initialize: {e}{RESET}")
            self.available = False
            return False

    # ------------------------------------------------------------------
    # Worker: collect sentences → synthesize as one block → play
    # ------------------------------------------------------------------

    def _speech_worker(self):
        """Background thread that collects queued sentences, then synthesizes
        and plays the collected text as a single continuous audio clip."""
        while self.running:
            try:
                if self.interrupt_event.is_set():
                    self.interrupt_event.clear()

                # Block until the first sentence arrives
                first = self.speech_queue.get(timeout=0.5)
                if first is None:
                    break
                if self.interrupt_event.is_set():
                    self.speech_queue.task_done()
                    continue

                # Collect more sentences that arrive within the timeout window
                collected = [first]
                self.speech_queue.task_done()

                while True:
                    try:
                        more = self.speech_queue.get(timeout=_COLLECT_TIMEOUT)
                        if more is None:
                            # Poison pill — synthesize what we have, then exit
                            collected_text = " ".join(collected)
                            if collected_text.strip():
                                self._playback_done.clear()
                                try:
                                    self._speak_text(collected_text)
                                finally:
                                    self._playback_done.set()
                            return
                        if more == _FLUSH_SENTINEL:
                            # Explicit flush — speak everything collected now
                            self.speech_queue.task_done()
                            break
                        if self.interrupt_event.is_set():
                            self.speech_queue.task_done()
                            break
                        collected.append(more)
                        self.speech_queue.task_done()
                    except queue.Empty:
                        # No more sentences within timeout — batch is complete
                        break

                if self.interrupt_event.is_set():
                    continue

                # Synthesize the full collected text as one clip
                full_text = " ".join(collected)
                if full_text.strip():
                    print(f"{CYAN}[TTS] Speaking {len(collected)} sentence(s): '{full_text[:80]}...'{RESET}" if len(full_text) > 80 else f"{CYAN}[TTS] Speaking {len(collected)} sentence(s): '{full_text}'{RESET}")
                    self._playback_done.clear()
                    try:
                        self._speak_text(full_text)
                    finally:
                        self._playback_done.set()

            except queue.Empty:
                continue

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove emojis, special characters, and markdown that shouldn't be spoken."""
        import emoji
        text = emoji.replace_emoji(text, replace='')
        # Remove markdown links [text](url) → text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        # Remove markdown formatting (* _ ~ # >)
        text = re.sub(r'[*_~`#>]', '', text)
        # Remove bullet points and numbered lists markers
        text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+[.)]\s+', '', text, flags=re.MULTILINE)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove parenthetical asides like (e.g., ...) that sound bad in TTS
        text = re.sub(r'\([^)]{0,50}\)', '', text)
        # Replace newlines with spaces
        text = re.sub(r'\n+', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _speak_text(self, text):
        """Synthesize and play text as one continuous audio clip."""
        text = self._clean_text(text)
        if not text or len(text) < 2 or not self._provider:
            return

        try:
            result = self._provider.synthesize(text, self.interrupt_event)
            if result is None or self.interrupt_event.is_set():
                return

            audio_data, samplerate = result
            self.current_playback = True
            sd.play(audio_data, samplerate=samplerate, blocking=True)
            self.current_playback = False

        except Exception as e:
            print(f"{YELLOW}[TTS Error]: {e}{RESET}")
            self.current_playback = False

    def queue_sentence(self, sentence):
        if self.enabled and self._initialized and sentence.strip():
            self._playback_done.clear()  # mark that there's pending work
            self.speech_queue.put(sentence)

    def flush_and_speak(self):
        """Signal the worker to speak everything collected so far.
        Call this after the LLM finishes streaming to avoid timeout delays."""
        if self.enabled and self._initialized:
            self.speech_queue.put(_FLUSH_SENTINEL)

    def stop(self):
        self.interrupt_event.set()
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()
        try:
            sd.stop()
        except:
            pass
        self.current_playback = False

    def wait_for_completion(self):
        """Wait until all queued text has been spoken."""
        if self.enabled:
            self.speech_queue.join()
            # Also wait for the actual audio playback to finish
            self._playback_done.wait()

    def toggle(self, enable):
        if enable and not self._initialized:
            if self.initialize():
                self.enabled = True
                return True
            return False
        self.enabled = enable
        return True

    def shutdown(self):
        self.running = False
        self.stop()
        self.speech_queue.put(None)


# Backward-compatible alias
PiperTTS = SmartTTS

# Global TTS instance
tts = SmartTTS()
