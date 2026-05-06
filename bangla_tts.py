"""
Bangla TTS Integration for LiveKit Agents
Uses Google Translate TTS (gTTS) optimized for Bangla
"""

import asyncio
import logging
from typing import Optional
import io
from livekit.agents.tts import TTS, SynthesizedAudio

logger = logging.getLogger("bangla_tts")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS not installed. Install with: pip install gTTS")

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


class BanglaTTS(TTS):
    """
    Bangla TTS using Google Translate TTS (gTTS)
    - Optimized specifically for Bangla language
    - Natural-sounding Bangla speech
    - Supports various Bangla text inputs
    """
    
    def __init__(self, *, language: str = "bn", slow: bool = False):
        """
        Initialize Bangla TTS
        
        Args:
            language: Language code (default: "bn" for Bangla)
            slow: If True, slower speech (better for complex text)
        """
        super().__init__()
        
        if not GTTS_AVAILABLE:
            raise ImportError(
                "gTTS not installed. "
                "Install with: pip install gTTS"
            )
        
        self.language = language
        self.slow = slow
        self._loop = None
        
        logger.info(f"Initializing Bangla TTS (language={language})")
    
    async def astart(self):
        """Start the TTS engine"""
        logger.debug("Starting Bangla TTS")
        self._loop = asyncio.get_event_loop()
        await super().astart()
    
    async def astop(self):
        """Stop the TTS engine"""
        logger.debug("Stopping Bangla TTS")
        await super().astop()
    
    async def synthesize(self, text: str) -> SynthesizedAudio:
        """
        Synthesize text to speech using gTTS
        
        Args:
            text: Text to synthesize (Bangla text)
            
        Returns:
            SynthesizedAudio with audio data
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return SynthesizedAudio(text="")
        
        try:
            logger.debug(f"Synthesizing Bangla text: {text[:100]}...")
            
            # Run gTTS in executor to avoid blocking
            def _synthesize():
                try:
                    # Create gTTS object for Bangla
                    tts = gTTS(text=text, lang=self.language, slow=self.slow)
                    
                    # Generate audio to bytes buffer
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    
                    audio_data = audio_buffer.read()
                    logger.debug(f"Generated {len(audio_data)} bytes of audio")
                    return audio_data
                except Exception as e:
                    logger.error(f"gTTS synthesis error: {e}")
                    raise
            
            audio_data = await self._loop.run_in_executor(None, _synthesize)
            
            # Return synthesized audio
            # gTTS produces MP3 by default
            return SynthesizedAudio(
                text=text,
                data=audio_data,
                mime_type="audio/mpeg"  # gTTS outputs MP3
            )
            
        except Exception as e:
            logger.error(f"Bangla TTS error: {e}", exc_info=True)
            raise


def create_bangla_tts(language: str = "bn", slow: bool = False) -> BanglaTTS:
    """
    Factory function to create Bangla TTS instance
    
    Args:
        language: Language code (default: "bn" for Bangla)
        slow: If True, slower speech for clarity
    
    Usage:
        tts = create_bangla_tts()              # Normal speed
        tts = create_bangla_tts(slow=True)     # Slower for clarity
    """
    return BanglaTTS(language=language, slow=slow)


class OpenAITTS(TTS):
    """
    TTS via OpenAI (if available). Falls back to None if OpenAI not configured.
    """
    def __init__(self, *, model: str = "gpt-4o-mini-tts", voice: str = None):
        super().__init__()
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        self.model = model
        self.voice = voice
        self._loop = None

    async def astart(self):
        self._loop = asyncio.get_event_loop()
        await super().astart()

    async def astop(self):
        await super().astop()

    async def synthesize(self, text: str) -> SynthesizedAudio:
        if not text or not text.strip():
            return SynthesizedAudio(text="")

        try:
            def _synth():
                try:
                    # Try OpenAI TTS API. API naming can vary by SDK version.
                    # Attempt a commonly used surface: openai.audio.speech.create
                    if hasattr(openai, "audio") and hasattr(openai.audio, "speech"):
                        out = openai.audio.speech.create(model=self.model, voice=self.voice, input=text)
                        # out could be bytes or a response object containing 'audio'
                        audio_bytes = getattr(out, "audio", None) or out.get("audio") or out
                        if isinstance(audio_bytes, bytes):
                            return audio_bytes
                    # Fallback: older SDKs expose an alternative
                    if hasattr(openai, "Audio"):
                        resp = openai.Audio.speech.create(model=self.model, text=text)
                        audio_bytes = getattr(resp, "audio", None) or resp.get("audio")
                        return audio_bytes
                    raise RuntimeError("OpenAI TTS API not available in installed openai package")
                except Exception as e:
                    logger.error(f"OpenAI TTS synthesis error: {e}")
                    raise

            audio_data = await asyncio.get_event_loop().run_in_executor(None, _synth)
            if not audio_data:
                raise RuntimeError("OpenAI TTS returned no audio data")

            return SynthesizedAudio(text=text, data=audio_data, mime_type="audio/mpeg")

        except Exception:
            logger.exception("OpenAI TTS failed, re-raising to caller")
            raise


def create_openai_tts(model: str = "gpt-4o-mini-tts", voice: str = None) -> OpenAITTS:
    return OpenAITTS(model=model, voice=voice)


# Convenience helper: synthesize text to an MP3 file (sync)
def synthesize_bangla_file(text: str, out_dir: str = None, filename: str = None, language: str = "bn", slow: bool = False) -> str:
    """Synthesize Bangla text to an MP3 file using gTTS and return the file path.

    This duplicates the small helper previously provided in `bangla_utils.py` so
    that TTS utilities live in one module.
    """
    import os
    import time

    if not GTTS_AVAILABLE:
        raise ImportError("gTTS not installed. Install with: pip install gTTS")

    if out_dir is None:
        out_dir = os.path.dirname(__file__)

    if filename is None:
        filename = f"bangla_tts_{int(time.time() * 1000)}.mp3"

    out_path = os.path.join(out_dir, filename)

    try:
        tts = gTTS(text=text, lang=language, slow=slow)
        tts.save(out_path)
        return out_path
    except Exception as e:
        logger.error(f"Failed to synthesize file: {e}")
        return ""


__all__ = [
    "BanglaTTS",
    "create_bangla_tts",
    "OpenAITTS",
    "create_openai_tts",
    "synthesize_bangla_file",
]
