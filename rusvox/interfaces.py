from typing import Any, Callable, Dict, List


class ASRModel:
    def __init__(self, transcribe_func: Callable[[List[Dict[str, Any]]], List[str]]):
        self.transcribe_func = transcribe_func

    def transcribe(self, audio_samples: List[Dict[str, Any]]) -> List[str]:
        return self.transcribe_func(audio_samples)


def create_custom_model(
    transcribe_func: Callable[[List[Dict[str, Any]]], List[str]],
) -> ASRModel:
    return ASRModel(transcribe_func)
