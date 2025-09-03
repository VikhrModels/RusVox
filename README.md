
# RusVox

RusVox - утилита оценки `ASR` моделей для [Russian ASR Leaderboard](https://huggingface.co/spaces/Vikhrmodels/Russian_ASR_Leaderboard). Данный репозиторий помогает легко адаптировать любой фреймворк для оценки на нескольких датасетах лидерборда.

## Установка

Советуем устанавливать библиотеку через `uv`. Как установить `uv` можно прочитать [здесь](https://docs.astral.sh/uv/getting-started/installation/).

1. Клонируйте репозиторий

```bash
git clone https://github.com/VikhrModels/RusVox
```

2. Перейдите в папку проекта

```bash
cd RusVox
```

3. Установите библиотеку

```bash
uv sync
```

4. Проверьте установку

```python
import rusvox
print(rusvox.__version__)
```

## Использование 

### Модель с поддержкой Hugging Face pipeline

```python
from rusvox import create_hf_model, run_evaluation, save_report

asr_model = create_hf_model("openai/whisper-large-v3")
report = run_evaluation(asr_model)
save_report(report)
```

### Кастомная моделька

```python
import tempfile
import soundfile as sf
from rusvox import create_custom_model, run_evaluation, save_report


model = ...  # Инициализируйте вашу ASR модель

def my_transcribe_func(audio_samples: list[dict[str, Any]]) -> list[str]:
    transcriptions = []
    for audio in audio_samples:
        audio_array = audio["array"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            sf.write(tmpfile.name, audio_array, audio["sampling_rate"])
            
            transcription = model.transcribe(tmpfile.name)
            preprocessed = preprocess_text(transcription)

            transcriptions.append(preprocessed)
    return transcriptions

asr_model = create_custom_model(my_transcribe_func)
report = run_evaluation(asr_model)
save_report(report, "my_report.json")
print(report)
```

## Описание датасетов

Датасеты, используемые в RusVox, представляют собой подмножества корпуса [RuASRBenchmark](https://huggingface.co/datasets/Vikhrmodels/RuASRBenchmark)

- **RuLS (Russian_LibriSpeech)**: 1352 записи
  Корпус на основе аудиокниг из проекта LibriVox, содержащих профессионально записанную русскоязычную речь. Подходит для оценки качества распознавания чистой, выразительной дикторской речи с верифицированными транскрипциями. Используется для тестирования моделей в условиях высокой разборчивости аудио.

- **CV 22.0 (Common_Voice_Corpus_22.0)**: 10244 записи  
  Многоязычный краудсорсинговый корпус от Mozilla Common Voice (версия 22.0), включающий русскоязычную речь от добровольцев. Характеризуется разнообразием дикторов, акцентов и условий записи. Применяется для проверки обобщающей способности моделей на реальных пользовательских записях с вариативным качеством.

- **Webinars (Tone_Webinars)**: 21587 записей
  Набор данных, собранный из образовательных вебинаров, содержит спонтанную речь с разнообразными дикторами, темами и стилями общения. Отражает реальные сценарии, такие как онлайн-лекции и обсуждения. Используется для оценки производительности моделей в условиях разговорной речи с переменным уровнем шума и перекрывающихся голосов.

- **Books (Tone_Books)**: 4930 записей
  Фрагменты русскоязычных аудиокниг с чистой дикторской речью и точными транскрипциями. Предназначен для тестирования моделей на высококачественных записях с литературным языком и четкой артикуляцией, что делает его идеальным для базовой оценки точности распознавания.

- **Speak (Tone_Speak)**: 700 записей.  
  Корпус синтетической русской речи, созданной с использованием технологий TTS. Позволяет оценить устойчивость моделей к искусственно сгенерированным голосам, которые часто используются в современных голосовых интерфейсах. Подходит для проверки способности моделей обрабатывать неестественные интонации и ритмы.

- **Sova (Sova_RuDevices)**: 5799 записей
  Набор данных с живой русской речью, записанной на различных устройствах с частотой дискретизации 16 кГц. Включает шумные условия, вариации качества микрофонов и спонтанную речь. Используется для тестирования робастности моделей к реальным сценариям, таким как голосовые ассистенты и мобильные приложения, где аудио может содержать фоновый шум и артефакты записи.
