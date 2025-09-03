from jiwer import wer, cer
import string
from .dataset import init_dataset
from .interfaces import ASRModel
from typing import Dict


def clear_text(text: str) -> str:
    text = text.strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def score_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    predictions = [clear_text(p) for p in predictions]
    references = [clear_text(r) for r in references]

    return {
        "wer": wer(reference=references, hypothesis=predictions),
        "cer": cer(reference=references, hypothesis=predictions),
    }


def run_evaluation(
    model: ASRModel, sample_rate: int = 16_000, num_workers: int = 2
) -> Dict[str, Dict[str, float]]:
    ds = init_dataset(sample_rate=sample_rate, num_workers=num_workers)
    report = {}
    for subset in ds.keys():
        subset_data = ds[subset]
        audio_samples = [s["audio"] for s in subset_data]
        references = [s["text"] for s in subset_data]
        predictions = model.transcribe(audio_samples)
        metrics = score_metrics(predictions, references)
        report[subset] = metrics
    return report
