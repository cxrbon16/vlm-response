import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from transformers import AutoProcessor
from datasets import load_dataset, load_from_disk
from PIL import Image
from vllm import LLM, SamplingParams

dataset = load_from_disk("data/refined_Berkesule_translated_mmiq_dataset_with_question_train.parquet")

example = dataset[10]

image = example["image"]

print(image)
print(type(image))

model_id = "Qwen/Qwen3-VL-8B-Thinking"  # veya "microsoft/phi-3-vision" da kullanılabilir
processor = AutoProcessor.from_pretrained(model_id)
llm = LLM(model=model_id, max_model_len=8192 * 4, limit_mm_per_prompt={"image": 1})
sampling_params = SamplingParams(
    max_tokens=40960, temperature=1.0, top_p=0.95, repetition_penalty=1.0, top_k=20,presence_penalty=0.0
)

input_dict = {
    "prompt": "<|vision_start|><|image_pad|><|vision_end|>Senden bu görseldeki problemi çözmeni istiyorum. Soru ve cevaplar görselde olacak.",
    "multi_modal_data":{
        "image": image,
    }
}

outputs = llm.generate(input_dict, sampling_params)

for o in outputs:
    print(o.outputs[0].text)
    print("----")
