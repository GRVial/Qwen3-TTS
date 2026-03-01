# Qwen3-TTS (Fork Privado)

Este repositório é um **fork privado** do projeto original Qwen3-TTS.

A referência completa continua em `qwentts_README.md`.

## O que mudou neste fork

- Pipeline local de pré-processamento para áudio de referência (modo Base):
  - Conversão/normalização com `ffmpeg` para mono, 16kHz, PCM s16
  - Redução de ruído com DeepFilterNet
  - Re-normalização final com `ffmpeg`
  - STT local com Faster-Whisper (CUDA + VAD)
- A UI agora gera automaticamente **preview de reference text** ao carregar áudio e preenche o campo de referência (editável antes de gerar).
- A UI exibe preview do áudio limpo, confiança média da transcrição, qualidade estimada e progresso por etapas (convertendo, limpando ruído, transcrevendo).
- O formato de exportação selecionável (`wav`, `mp3`, `ogg`, `m4a`) é aplicado ao **Output Audio final**.
- Fluxo permanece 100% local/offline.

## Setup rápido

```bash
# instalar este fork em modo editável
pip install -e .
```

## Nota de compatibilidade (DeepFilterNet)

Em alguns ambientes, pode ocorrer conflito entre `deepfilternet` e `wheel` por causa da versão de `packaging`.

Se aparecer erro de dependência, rode:

```bash
python -m pip uninstall -y wheel
python -m pip install "packaging>=23,<24"
python -m pip check
```

Se o `pip check` retornar `No broken requirements found.`, o ambiente está consistente.

### (Opcional) FlashAttention 2

```bash
pip install -U flash-attn --no-build-isolation
```

Em máquinas com pouca RAM:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

## Comandos de execução (Web UI)

```bash
# ajuda
qwen-tts-demo --help
```

### CustomVoice

```bash
# com FlashAttention (padrão)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000

# sem FlashAttention
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000 --no-flash-attn
```

### VoiceDesign

```bash
# com FlashAttention (padrão)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --ip 0.0.0.0 --port 8000

# sem FlashAttention
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --ip 0.0.0.0 --port 8000 --no-flash-attn
```

### Base

```bash
# com FlashAttention (padrão)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000

# sem FlashAttention
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000 --no-flash-attn
```

Depois, abra no navegador:

- `http://<seu-ip>:8000`

---

Se quiser todas as instruções detalhadas (API Python, tokenizer, fine-tuning, vLLM, HTTPS, etc.), consulte `qwentts_README.md`.
