import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm.auto import tqdm

BATCH_SIZE = 10000
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT = """
Ты - строгий корректор русского текста. Твоя единственная задача - исправить данный текст следующим образом:
1. Привести все числа, записанные цифрами, в буквенный вид (например, 7 -> семь).
Не вноси никаких других изменений: не меняй слова, не добавляй или не удаляй контент, не исправляй орфографию или пунктуацию. Выводи ТОЛЬКО исправленный текст без каких-либо объяснений, мыслей, шагов или дополнительного содержимого. Если текст уже идеален, верни его без изменений.
"""


def correct_texts(texts: list[str]) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=1024,
        dtype="auto",
    )
    sampling_params = SamplingParams(
        max_tokens=500,
        temperature=0.70,
        top_p=0.80,
        top_k=20,
        min_p=0.0,
    )

    corrected_texts = []
    for i in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Обработка примеров",
    ):
        batch = texts[i : i + BATCH_SIZE]
        messages_list = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
            for text in batch
        ]
        prompts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_list
        ]
        outputs = llm.generate(prompts, sampling_params)
        corrected_batch = [output.outputs[0].text.strip() for output in outputs]
        corrected_texts.extend(corrected_batch)

    del llm
    torch.cuda.empty_cache()

    return corrected_texts
