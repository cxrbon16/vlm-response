from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
import torch

# 🔹 1. Dataset'i yükle
dataset = load_from_disk("data/refined_Berkesule_translated_mmiq_dataset_with_question_train.parquet")

# train split'ten ilk örneği al
example = dataset[0]

# Görseli doğrudan al (PIL Image)
image = example["image"]

# 🔹 2. Modeli ve processor'u yükle (örnek: LLaVA 1.5)
model_id = "Qwen/Qwen3-VL-4B-Thinking"  # veya "microsoft/phi-3-vision" da kullanılabilir

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)


# 🔹 3. Prompt hazırla
prompt = f"USER: I want you to solve this problem in the image. Question and answers will be both in the image. \nASSISTANT:"

# 🔹 4. Görsel + text birlikte encode et

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": "I want you to solve this problem in the image. Question and answers will be both in the image."}
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# 🔹 5. Text generate et

generated_ids = model.generate(**inputs, max_new_tokens=40960, temperature=1.0, top_p=0.95, repetition_penalty=1.0, use_cache=True, top_k=20)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
