import threading
from dataclasses import dataclass
from importlib import import_module
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TranscriptionResult:
    text: str
    avg_confidence: float


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._lock = threading.Lock()
        self._model = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                whisper_mod = import_module("faster_whisper")
            except Exception as exc:
                raise RuntimeError(
                    "faster-whisper is not available. Install package `faster-whisper` for local transcription."
                ) from exc

            WhisperModel = getattr(whisper_mod, "WhisperModel")
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )

    @staticmethod
    def _segment_confidence(segment: object) -> float:
        logp = getattr(segment, "avg_logprob", None)
        if logp is None:
            return 0.0
        confidence = float(np.exp(float(logp)))
        return float(min(1.0, max(0.0, confidence)))

    def transcribe_file(
        self, input_path: str, language: Optional[str] = None, vad_filter: bool = True
    ) -> TranscriptionResult:
        self._lazy_init()
        assert self._model is not None

        seg_iter, _info = self._model.transcribe(
            input_path,
            beam_size=self.beam_size,
            vad_filter=vad_filter,
            language=language,
            condition_on_previous_text=False,
        )

        texts: List[str] = []
        confs: List[float] = []
        for segment in seg_iter:
            text = (getattr(segment, "text", "") or "").strip()
            if text:
                texts.append(text)
                confs.append(self._segment_confidence(segment))

        merged = " ".join(texts).strip()
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return TranscriptionResult(text=merged, avg_confidence=avg_conf)


faster_whisper_transcriber = FasterWhisperTranscriber()
