"""
BanglaSpeech2Text STT Integration for LiveKit Agents
Offline Bangla speech-to-text using fine-tuned Whisper model
"""

import asyncio
import logging
from typing import Optional
import numpy as np
from livekit.agents.stt import STT, SpeechData, SpeechEvent, SpeechEventType
import io
import wave

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

logger = logging.getLogger("bangla_stt")

try:
    from banglaspeech2text import Speech2Text
    BANGLA_SPEECH_AVAILABLE = True
except ImportError:
    BANGLA_SPEECH_AVAILABLE = False
    logger.warning("BanglaSpeech2Text not installed. Install with: pip install banglaspeech2text")


class BanglaSpeechSTT(STT):
    """
    Offline STT using BanglaSpeech2Text (fine-tuned Whisper for Bangla)
    - Offline processing (no API calls)
    - Optimized for Bangla phonetics
    - Zero cost, lower latency
    
    Models available:
    - "tiny": ~100-200MB, WER=74 (fastest)
    - "base": ~200-300MB, WER=46 (default, good balance)
    - "small": ~1GB, WER=18 (better quality)
    - "large": ~3-4GB, WER=11 (best quality, slower)
    """
    
    def __init__(self, *, model: str = "base"):
        """
        Initialize BanglaSpeech2Text STT
        
        Args:
            model: Model size ("tiny", "base", "small", "large")
        """
        super().__init__()
        
        if not BANGLA_SPEECH_AVAILABLE:
            raise ImportError(
                "BanglaSpeech2Text not installed. "
                "Install with: pip install banglaspeech2text"
            )
        
        self.model_name = model
        self._recognizer = None
        self._loop = None
        
        logger.info(f"Initializing BanglaSpeech2Text STT (model={model})")
    
    async def astart(self):
        """Start the STT engine asynchronously"""
        logger.debug(f"Starting BanglaSpeech2Text STT with model={self.model_name}")
        
        self._loop = asyncio.get_event_loop()
        
        def _load_model():
            try:
                # Initialize Speech2Text with selected model
                logger.info(f"Loading BanglaSpeech2Text {self.model_name} model...")
                recognizer = Speech2Text(self.model_name)
                logger.info(f"BanglaSpeech2Text {self.model_name} model loaded successfully")
                return recognizer
            except Exception as e:
                logger.error(f"Failed to load BanglaSpeech2Text model: {e}")
                raise
        
        # Run model loading in executor to avoid blocking
        self._recognizer = await self._loop.run_in_executor(None, _load_model)
        await super().astart()
    
    async def astop(self):
        """Stop the STT engine"""
        logger.debug("Stopping BanglaSpeech2Text STT")
        await super().astop()
        self._recognizer = None
    
    async def recognize(self, audio: SpeechData) -> Optional[SpeechEvent]:
        """
        Transcribe audio using BanglaSpeech2Text
        
        Args:
            audio: SpeechData object containing audio frames
            
        Returns:
            SpeechEvent with transcription result
        """
        if self._recognizer is None:
            logger.error("STT not initialized. Call astart() first.")
            return None
        
        try:
            # Collect audio frames into a single buffer
            audio_buffer = np.array([], dtype=np.float32)
            
            for frame in audio.frames:
                # Convert frame bytes to numpy array (audio frames are 16-bit PCM)
                frame_data = np.frombuffer(frame.data, dtype=np.float32)
                audio_buffer = np.concatenate([audio_buffer, frame_data])
            
            if len(audio_buffer) == 0:
                logger.debug("Empty audio buffer received")
                return None
            
            logger.debug(f"Processing {len(audio_buffer)} audio samples (~{len(audio_buffer)/16000:.1f}s)")
            
            # Run transcription in executor (non-blocking)
            def _transcribe():
                try:
                    # BanglaSpeech2Text.recognize() accepts numpy arrays
                    result = self._recognizer.recognize(audio_buffer)
                    return result
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    raise
            
            result = await self._loop.run_in_executor(None, _transcribe)
            
            if result:
                text = str(result).strip()
                if text:
                    logger.info(f"Transcribed: {text}")
                    
                    return SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[text],
                    )
                else:
                    logger.debug("Empty transcription result")
                    return None
            else:
                logger.debug("No transcription result")
                return None
                
        except Exception as e:
            logger.error(f"Recognition error: {e}", exc_info=True)
            return None


# Factory function for easy integration
def create_bangla_stt(model: str = "base") -> BanglaSpeechSTT:
    """
    Factory function to create BanglaSpeech2Text STT instance
    
    Args:
        model: Model size ("tiny"=fastest, "base"=recommended, "small", "large"=best)
    
    Usage:
        stt = create_bangla_stt()              # Uses "base" model
        stt = create_bangla_stt(model="small") # Better quality
    """
    return BanglaSpeechSTT(model=model)


class OpenAIWhisperSTT(STT):
    """
    STT using OpenAI Whisper API (cloud). Falls back to None if OpenAI not configured.
    """

    def __init__(self, *, model: str = "whisper-1"):
        super().__init__()
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        self.model = model
        self._loop = None

    async def astart(self):
        self._loop = asyncio.get_event_loop()
        await super().astart()

    async def astop(self):
        await super().astop()

    async def recognize(self, audio: SpeechData) -> Optional[SpeechEvent]:
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        # collect frames
        try:
            float_buffer = np.array([], dtype=np.float32)
            for frame in audio.frames:
                frame_data = np.frombuffer(frame.data, dtype=np.float32)
                float_buffer = np.concatenate([float_buffer, frame_data])

            if float_buffer.size == 0:
                return None

            # Convert float32 [-1,1] to int16 PCM
            int16 = (np.clip(float_buffer, -1.0, 1.0) * 32767).astype(np.int16)
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(int16.tobytes())

            wav_io.seek(0)

            def _transcribe():
                try:
                    # Use OpenAI Whisper API
                    # Note: API surface can change; this uses the common pattern openai.Audio.transcribe
                    resp = openai.Audio.transcribe(model=self.model, file=wav_io)
                    text = getattr(resp, "text", None) or resp.get("text")
                    return text
                except Exception as e:
                    logger.error(f"OpenAI transcription failed: {e}")
                    raise

            text = await self._loop.run_in_executor(None, _transcribe)
            if text:
                return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[str(text).strip()])
            return None

        except Exception as e:
            logger.error("OpenAI recognition error", exc_info=True)
            return None


def create_openai_stt(model: str = "whisper-1") -> OpenAIWhisperSTT:
    return OpenAIWhisperSTT(model=model)


