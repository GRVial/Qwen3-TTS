import threading
import importlib.util
import sys
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass
class _CompatAudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def _install_torchaudio_backend_compat() -> None:
    try:
        import torchaudio as ta
    except Exception:
        return

    try:
        import soundfile as sf
    except Exception:
        sf = None

    if sf is not None:

        def _ta_load(path: str, frame_offset: int = 0, num_frames: int = -1, **_kwargs):
            start = int(frame_offset or 0)
            frames = -1 if num_frames is None else int(num_frames)
            if frames == 0:
                frames = -1
            data, sample_rate = sf.read(
                path, start=start, frames=frames, dtype="float32", always_2d=True
            )
            import torch

            audio = torch.from_numpy(data.T.copy())
            return audio, int(sample_rate)

        def _ta_save(path: str, src, sample_rate: int, **_kwargs):
            import numpy as np
            import torch

            audio = (
                src.detach().cpu().numpy() if torch.is_tensor(src) else np.asarray(src)
            )
            if audio.ndim == 1:
                audio = audio[None, :]
            sf.write(path, audio.T, int(sample_rate))

        setattr(ta, "load", _ta_load)
        setattr(ta, "save", _ta_save)

    if not hasattr(ta, "info"):
        if sf is not None:

            def _ta_info(path: str, **_kwargs):
                i = sf.info(path)
                subtype = str(getattr(i, "subtype", "") or "")
                bits = 0
                if "PCM_16" in subtype:
                    bits = 16
                elif "PCM_24" in subtype:
                    bits = 24
                elif "PCM_32" in subtype:
                    bits = 32
                return _CompatAudioMetaData(
                    sample_rate=int(i.samplerate),
                    num_frames=int(i.frames),
                    num_channels=int(i.channels),
                    bits_per_sample=int(bits),
                    encoding=str(getattr(i, "format", "UNKNOWN") or "UNKNOWN"),
                )

            setattr(ta, "info", _ta_info)

    has_backend_common = False
    try:
        has_backend_common = (
            importlib.util.find_spec("torchaudio.backend.common") is not None
        )
    except ModuleNotFoundError:
        has_backend_common = False

    if not has_backend_common:
        backend_mod = sys.modules.get("torchaudio.backend")
        if backend_mod is None:
            backend_mod = ModuleType("torchaudio.backend")
            sys.modules["torchaudio.backend"] = backend_mod

        common_mod = sys.modules.get("torchaudio.backend.common")
        if common_mod is None:
            common_mod = ModuleType("torchaudio.backend.common")
            sys.modules["torchaudio.backend.common"] = common_mod

        setattr(common_mod, "AudioMetaData", _CompatAudioMetaData)
        setattr(backend_mod, "common", common_mod)


class DeepFilterNetReducer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._model = None
        self._state = None
        self._enhance = None
        self._load_audio = None
        self._save_audio = None

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            _install_torchaudio_backend_compat()
            try:
                enhance_mod = import_module("df.enhance")
            except Exception as exc:
                raise RuntimeError(
                    "DeepFilterNet is not available in this environment. "
                    "Install package `deepfilternet` and ensure dependencies (e.g. torchaudio backend) are working. "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc

            enhance = getattr(enhance_mod, "enhance")
            init_df = getattr(enhance_mod, "init_df")
            load_audio = getattr(enhance_mod, "load_audio")
            save_audio = getattr(enhance_mod, "save_audio")

            model, state, _ = init_df()
            self._model = model
            self._state = state
            self._enhance = enhance
            self._load_audio = load_audio
            self._save_audio = save_audio
            self._initialized = True

    def is_available(self) -> bool:
        try:
            self._lazy_init()
            return True
        except Exception:
            return False

    def denoise_file(self, input_path: str, output_path: str) -> str:
        self._lazy_init()

        assert self._state is not None
        assert self._enhance is not None
        assert self._load_audio is not None
        assert self._save_audio is not None

        audio, _ = self._load_audio(input_path, sr=self._state.sr())
        enhanced = self._enhance(self._model, self._state, audio)
        self._save_audio(output_path, enhanced, self._state.sr())
        return output_path


deepfilternet_reducer = DeepFilterNetReducer()
