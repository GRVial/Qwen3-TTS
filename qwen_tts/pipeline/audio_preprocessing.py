import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

from .noise_reduction import deepfilternet_reducer
from .transcription import faster_whisper_transcriber

logger = logging.getLogger(__name__)


StageCallback = Optional[Callable[[str, str], None]]


@dataclass
class AudioPipelineConfig:
    ffmpeg_bin: str = "ffmpeg"
    timeout_seconds: int = 120
    remove_silence: bool = False
    use_cache: bool = True
    whisper_language: Optional[str] = None


@dataclass
class AudioPipelineResult:
    clean_audio: Tuple[np.ndarray, int]
    clean_audio_path: str
    transcript_text: str
    transcript_confidence: float
    quality_score: float
    denoise_backend: str
    denoise_warning: Optional[str] = None


class AudioPreprocessingPipeline:
    def __init__(
        self, config: Optional[AudioPipelineConfig] = None, max_workers: int = 2
    ) -> None:
        self.config = config or AudioPipelineConfig()
        self._cache: Dict[str, AudioPipelineResult] = {}
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="audio-pipeline"
        )

        self._ensure_ffmpeg()

    def _ensure_ffmpeg(self) -> None:
        if shutil.which(self.config.ffmpeg_bin) is None:
            raise RuntimeError(
                f"ffmpeg was not found in PATH. Please install ffmpeg or configure ffmpeg_bin='{self.config.ffmpeg_bin}'."
            )

    @staticmethod
    def _log_event(event: str, **kwargs: object) -> None:
        payload = {"event": event, **kwargs}
        logger.info(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _to_float_audio(data: np.ndarray) -> np.ndarray:
        x = np.asarray(data)
        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            denom = max(abs(info.min), info.max)
            y = x.astype(np.float32) / max(1, float(denom))
        elif np.issubdtype(x.dtype, np.floating):
            y = x.astype(np.float32)
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            if peak > 1.0:
                y = y / peak
        else:
            raise TypeError(f"Unsupported dtype: {x.dtype}")

        if y.ndim > 1:
            y = y.mean(axis=-1)
        return np.clip(y, -1.0, 1.0)

    @staticmethod
    def _hash_audio(audio: Tuple[np.ndarray, int]) -> str:
        wav, sr = audio
        h = hashlib.sha256()
        h.update(str(int(sr)).encode("utf-8"))
        h.update(np.asarray(wav, dtype=np.float32).tobytes())
        return h.hexdigest()

    def _run_cmd(self, cmd: list[str], timeout: int) -> None:
        started = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = round((time.time() - started) * 1000)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{stderr}")
        self._log_event("ffmpeg_ok", elapsed_ms=elapsed, cmd=" ".join(cmd))

    def _to_temp_wav(self, audio_input: object, workdir: str) -> str:
        if isinstance(audio_input, str):
            if not os.path.isfile(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            return audio_input

        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            first, second = audio_input
            if isinstance(first, int):
                sr = int(first)
                wav = np.asarray(second)
            else:
                wav = np.asarray(first)
                sr = int(second)
            wav = self._to_float_audio(wav)
            src_path = os.path.join(workdir, "input_tuple.wav")
            sf.write(src_path, wav, sr)
            return src_path

        if (
            isinstance(audio_input, dict)
            and "sampling_rate" in audio_input
            and "data" in audio_input
        ):
            sr = int(audio_input["sampling_rate"])
            wav = self._to_float_audio(np.asarray(audio_input["data"]))
            src_path = os.path.join(workdir, "input_dict.wav")
            sf.write(src_path, wav, sr)
            return src_path

        raise ValueError(
            "Unsupported audio input. Provide a file path, (sr, wav)/(wav, sr), or gradio audio dict."
        )

    def _convert_initial(self, src_path: str, dst_path: str) -> None:
        silence_filter = ""
        if self.config.remove_silence:
            silence_filter = ",silenceremove=start_periods=1:start_silence=0.1:start_threshold=-45dB:stop_periods=-1:stop_duration=0.2:stop_threshold=-45dB"

        afilter = f"loudnorm=I=-18:LRA=7:TP=-2{silence_filter}"

        cmd = [
            self.config.ffmpeg_bin,
            "-y",
            "-i",
            src_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            "-af",
            afilter,
            dst_path,
        ]
        self._run_cmd(cmd, timeout=self.config.timeout_seconds)

    def _normalize_final(self, src_path: str, dst_path: str) -> None:
        cmd = [
            self.config.ffmpeg_bin,
            "-y",
            "-i",
            src_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            "-af",
            "loudnorm=I=-18:LRA=7:TP=-2",
            dst_path,
        ]
        self._run_cmd(cmd, timeout=self.config.timeout_seconds)

    def _ffmpeg_denoise_fallback(self, src_path: str, dst_path: str) -> None:
        cmd = [
            self.config.ffmpeg_bin,
            "-y",
            "-i",
            src_path,
            "-af",
            "afftdn=nf=-25",
            dst_path,
        ]
        self._run_cmd(cmd, timeout=self.config.timeout_seconds)

    @staticmethod
    def _estimate_quality(
        wav: np.ndarray, sr: int, transcript_confidence: float
    ) -> float:
        x = np.asarray(wav, dtype=np.float32)
        if x.size == 0:
            return 0.0

        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        dbfs = 20 * np.log10(max(rms, 1e-8))
        clipping = float(np.mean(np.abs(x) >= 0.995))
        duration_s = x.shape[0] / max(1, sr)

        loudness_score = 1.0 - min(1.0, abs(dbfs + 20.0) / 30.0)
        clip_penalty = min(1.0, clipping * 10.0)
        duration_score = min(1.0, duration_s / 4.0)

        score = 100.0 * (
            0.40 * loudness_score
            + 0.35 * (1.0 - clip_penalty)
            + 0.25 * transcript_confidence * duration_score
        )
        return float(max(0.0, min(100.0, score)))

    def _process_impl(
        self, audio_input: object, callback: StageCallback = None
    ) -> AudioPipelineResult:
        with tempfile.TemporaryDirectory(prefix="qwen_tts_preproc_") as workdir:
            src_path = self._to_temp_wav(audio_input, workdir)

            if callback:
                callback("convert", "Convertendo e padronizando áudio...")
            converted_path = os.path.join(workdir, "01_converted.wav")
            self._convert_initial(src_path, converted_path)

            if callback:
                callback("denoise", "Limpando ruído com DeepFilterNet...")
            denoised_path = os.path.join(workdir, "02_denoised.wav")
            denoise_backend = "deepfilternet"
            denoise_warning: Optional[str] = None
            try:
                deepfilternet_reducer.denoise_file(converted_path, denoised_path)
            except Exception as exc:
                denoise_backend = "ffmpeg_afftdn_fallback"
                denoise_warning = (
                    "DeepFilterNet falhou; usando fallback de denoise com ffmpeg (afftdn). "
                    f"Motivo: {type(exc).__name__}: {exc}"
                )
                self._log_event(
                    "denoise_fallback",
                    backend=denoise_backend,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if callback:
                    callback(
                        "denoise",
                        "DeepFilterNet indisponível; aplicando fallback de denoise com ffmpeg...",
                    )
                self._ffmpeg_denoise_fallback(converted_path, denoised_path)

            if callback:
                callback("normalize", "Re-normalizando áudio final...")
            final_path = os.path.join(workdir, "03_final_16k_mono_s16.wav")
            self._normalize_final(denoised_path, final_path)

            if callback:
                callback("stt", "Transcrevendo com Faster-Whisper (CUDA + VAD)...")
            tx = faster_whisper_transcriber.transcribe_file(
                final_path,
                language=self.config.whisper_language,
                vad_filter=True,
            )

            wav, sr = sf.read(final_path, dtype="float32")
            wav = self._to_float_audio(wav)
            clean_audio = (wav, int(sr))
            quality_score = self._estimate_quality(wav, int(sr), tx.avg_confidence)

            export_path = os.path.join(
                tempfile.gettempdir(), f"qwen_tts_clean_{int(time.time() * 1000)}.wav"
            )
            sf.write(export_path, wav, int(sr), subtype="PCM_16")

            return AudioPipelineResult(
                clean_audio=clean_audio,
                clean_audio_path=export_path,
                transcript_text=tx.text,
                transcript_confidence=tx.avg_confidence,
                quality_score=quality_score,
                denoise_backend=denoise_backend,
                denoise_warning=denoise_warning,
            )

    def process(
        self, audio_input: object, callback: StageCallback = None
    ) -> AudioPipelineResult:
        cache_key = None

        if self.config.use_cache:
            try:
                if isinstance(audio_input, tuple) and len(audio_input) == 2:
                    if isinstance(audio_input[0], int):
                        cache_key = self._hash_audio(
                            (self._to_float_audio(audio_input[1]), int(audio_input[0]))
                        )
                    else:
                        cache_key = self._hash_audio(
                            (self._to_float_audio(audio_input[0]), int(audio_input[1]))
                        )
            except Exception:
                cache_key = None

        if cache_key is not None:
            with self._cache_lock:
                hit = self._cache.get(cache_key)
            if hit is not None:
                self._log_event("cache_hit", key=cache_key)
                return hit

        future = self._executor.submit(self._process_impl, audio_input, callback)
        try:
            result = future.result(timeout=self.config.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Audio preprocessing timeout after {self.config.timeout_seconds}s."
            ) from exc

        if cache_key is not None:
            with self._cache_lock:
                self._cache[cache_key] = result

        return result
