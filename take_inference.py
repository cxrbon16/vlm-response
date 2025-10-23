from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# 🔹 1. Dataset'i yükle
dataset = load_dataset("data/refined_Berkesule_translated_mmiq_dataset_with_question_train.parquet")

# train split'ten ilk örneği al
example = dataset["train"][0]

# Görseli doğrudan al (PIL Image)
image = example["image"]

# 🔹 2. Modeli ve processor'u yükle (örnek: LLaVA 1.5)
model_id = "Qwen/Qwen3-VL-4B-Thinking"  # veya "microsoft/phi-3-vision" da kullanılabilir

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

# 🔹 3. Prompt hazırla
prompt = f"USER: I want you to solve this problem in the image. Question and answers will be both in the image. \nASSISTANT:"

# 🔹 4. Görsel + text birlikte encode et
inputs = processor(prompt, image, return_tensors="pt").to("cuda")

# 🔹 5. Text generate et
output = model.generate(**inputs, max_new_tokens=100)
generated_text = processor.decode(output[0], skip_special_tokens=True)

print("🖼️ Question:", prompt)
print("💬 Model Output:", generated_text)