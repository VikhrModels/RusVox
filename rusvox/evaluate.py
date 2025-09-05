from jiwer import wer, cer
import string
from typing import Dict, List
from tqdm import tqdm
import json

from .dataset import init_dataset
from .interfaces import ASRModel


def clear_text(text: str) -> str:
    """Очищает текст для расчёта метрик: strip, lower, remove punctuation."""
    text = text.strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def score_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Расчитывает WER и CER на очищенных предсказаниях и референсах."""
    predictions = [clear_text(p) for p in predictions]
    references = [clear_text(r) for r in references]

    return {
        "wer": wer(reference=references, hypothesis=predictions),
        "cer": cer(reference=references, hypothesis=predictions),
    }


def run_evaluation(
    model: ASRModel, sample_rate: int = 16_000, num_workers: int = 2
) -> Dict[str, Dict[str, float]]:
    """Запускает оценку модели на всех subsets датасета с прогресс-баром."""
    ds = init_dataset(target_sr=sample_rate, num_workers=num_workers)
    report = {}
    for subset in tqdm(ds.keys(), desc="Evaluating subsets"):
        subset_data = ds[subset]
        audio_samples = [s["audio"] for s in subset_data]
        references = [s["text"] for s in subset_data]
        predictions = model.transcribe(audio_samples)
        metrics = score_metrics(predictions, references)
        report[subset] = metrics
    return report


def save_report(report: Dict[str, Dict[str, float]], path: str = "report.json") -> None:
    """Сохраняет отчёт в JSON файл."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
