from datasets import load_dataset, Audio


def init_dataset(num_workers: int = 2, target_sr: int = 16_000):
    """Инициализирует датасет RuASRBenchmark с приведением аудио к target_sr."""
    ds = load_dataset("Vikhrmodels/RuASRBenchmark", num_workers=num_workers)

    for split in ds.keys():
        ds[split] = ds[split].cast_column("audio", Audio(sampling_rate=target_sr))

    return ds
