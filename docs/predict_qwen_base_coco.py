import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoProcessor, BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

# === 参数设置 ===
prompt = "Please describe the image in detail, using no more than 50 words."
MAX_LENGTH = 512
MAX_SAMPLES = 500
VAL_JSON_PATH = "coco/coco_val_conversation.json"
LOCAL_MODEL_PATH = "D:/Project/qwen_instruct_model/Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_CSV = "./coco/eval_base_only.csv"
OUTPUT_PLOT = "./coco/eval_base_only_plot.png"

# === 加载 tokenizer / processor / embedder ===
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True, local_files_only=True)
processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True, local_files_only=True)
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
smoothie = SmoothingFunction().method4

# === 加载模型 ===
print("Loading base model...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()

# === 加载验证数据 ===
with open(VAL_JSON_PATH, "r") as f:
    val_data = json.load(f)
val_data = val_data[:MAX_SAMPLES]

# === 推理函数 ===
def predict(messages, model):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# === 主评估流程 ===
results = []
print("Running base model evaluation...")

for sample in tqdm(val_data):
    image_path = sample["conversations"][0]["value"]
    gt_text = sample["conversations"][1]["value"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path, "resized_height": 300, "resized_width": 300},
            {"type": "text", "text": prompt}
        ]
    }]

    pred_base = predict(messages, base_model)

    emb_gt = embedder.encode(gt_text)
    emb_base = embedder.encode(pred_base)
    sim_base = cosine_similarity([emb_gt], [emb_base])[0][0]

    bleu_base = sentence_bleu([gt_text.split()], pred_base.split(), smoothing_function=smoothie)
    _, _, f1_base = bertscore([pred_base], [gt_text], lang="en", rescale_with_baseline=True)

    results.append({
        "image_path": image_path,
        "gt": gt_text,
        "base_pred": pred_base,
        "bleu_base": bleu_base,
        "bertscore_base": f1_base[0].item(),
        "semantic_base": sim_base,
    })

# === 保存结果 ===
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Results saved to {OUTPUT_CSV}")

# === 可视化 ===
x = list(range(len(df)))
plt.figure(figsize=(10, 6))
plt.plot(x, df["bertscore_base"], label="BERTScore Base", color="red", linestyle="-")
plt.plot(x, df["bleu_base"], label="BLEU Base", color="grey", linestyle="--")
plt.xlabel("Sample Index")
plt.ylabel("Score")
plt.title("Base Model Evaluation")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.show()

print("\n=========== Summary ===========")
print(f"Avg BLEU       : {df['bleu_base'].mean():.4f}")
print(f"Avg BERTScore  : {df['bertscore_base'].mean():.4f}")
print(f"Avg Similarity : {df['semantic_base'].mean():.4f}")
