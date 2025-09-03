from typing import Any, Callable, Dict, List
import tempfile
import soundfile as sf
from transformers import pipeline
import torch

def create_custom_model(
    transcribe_func: Callable[[List[Dict[str, Any]]], List[str]],
) -> "ASRModel":
    """Создаёт ASR модель из кастомной функции транскрипции."""
    return ASRModel(transcribe_func)

def create_hf_model(
    model_id: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> "ASRModel":
    """Создаёт ASR модель на основе Hugging Face pipeline с fallback на временные файлы."""
    

    pipe = pipeline("automatic-speech-recognition", model=model_id, device=device)

    def hf_transcribe(audio_samples: List[Dict[str, Any]]) -> List[str]:
        results = []
        for audio in audio_samples:
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]
            if audio_array is None or len(audio_array) == 0:
                results.append("")
                continue
            try:
                result = pipe(audio_array)["text"]
            except Exception:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                    sf.write(tmpfile.name, audio_array, sampling_rate)
                    result = pipe(tmpfile.name)["text"]
            results.append(result)
        return results

    return ASRModel(hf_transcribe)

class ASRModel:
    def __init__(self, transcribe_func: Callable[[List[Dict[str, Any]]], List[str]]):
        self.transcribe_func = transcribe_func

    def transcribe(self, audio_samples: List[Dict[str, Any]]) -> List[str]:
        """Выполняет транскрипцию на списке аудио-сэмплов."""
        return self.transcribe_func(audio_samples)