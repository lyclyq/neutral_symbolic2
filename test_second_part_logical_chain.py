import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import pandas as pd
from peft import LoraConfig, get_peft_model
import numpy as np
# === configuration ===
model_dir = "./output/Qwen2-VL-2B-Step2/checkpoint-560"  #  LoRA checkpoint 
val_dataset_json_path = "latex_ocr_val_for_bce.json"
disease_list = [
    "atelectasis", "cardiomegaly", "consolidation", "edema", "effusion",
    "emphysema", "fibrosis", "hernia", "infiltration", "mass", "no finding",
    "nodule", "pleural thickening", "pneumonia", "pneumothorax"
]

# === load classifier (same as training) ===
class QwenWithClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        hidden_size = base_model.config.hidden_size
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 15)
        )
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None, **kwargs):
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                **kwargs
            )
        hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.classification_head(pooled)

# === load model and LoRA ===
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True,
)

# === 加载与训练一致的 LoRA 配置 ===
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=12,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="lora_only"
)
base_model = get_peft_model(base_model, lora_config)



# === 构建带分类头的结构 ===
model = QwenWithClassifier(base_model)
# === 加载 LoRA + 分类头权重（不包含 Qwen 主体）===
weights_path = "./output/Qwen2-VL-2B-Step2/classifier_only/pytorch_model.bin"  # ← 改成你保存的路径
state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval().cuda()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# === preprocess function ===
prompt = "What abnormalities are shown in this chest X-ray image?"
def preprocess(example):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": example["image_path"], "resized_height": 350, "resized_width": 350},
            {"type": "text", "text": prompt},
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    print('process_finshed')
    return {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "pixel_values": inputs["pixel_values"].squeeze(0),
        "image_grid_thw": inputs["image_grid_thw"][0],
        "labels": torch.tensor(example["disease_labels"], dtype=torch.float32)
    }


# === load validation set ===
val_ds = Dataset.from_json(val_dataset_json_path)
val_ds = val_ds.select(range(200))
val_ds = val_ds.map(preprocess)
val_ds.set_format(type="torch")

# === inference and evaluation ===


# === 多阈值评估 ===
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
metrics_list = []

for threshold in thresholds:
    TP = FP = FN = TN = 0
    print(f"\n=== Evaluating at threshold: {threshold:.2f} ===")

    for example in val_ds:
        for k in ["input_ids", "attention_mask", "pixel_values"]:
            example[k] = example[k].unsqueeze(0).cuda()
        example["image_grid_thw"] = example["image_grid_thw"].unsqueeze(0).cuda()

        inputs = {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "pixel_values": example["pixel_values"],
            "image_grid_thw": example["image_grid_thw"],
        }

        with torch.no_grad():
            logits = model(**inputs).cpu()
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()

        gt = example["labels"].int()

        for i in range(15):
            p, g = preds[0][i], gt[i]
            if p == 1 and g == 1: TP += 1
            elif p == 1 and g == 0: FP += 1
            elif p == 0 and g == 1: FN += 1
            elif p == 0 and g == 0: TN += 1

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

    metrics_list.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    })

# === 保存结果到 CSV ===
df_metrics = pd.DataFrame(metrics_list)
df_metrics.to_csv("threshold_search_results.csv", index=False)
print("所有阈值的指标已保存至 threshold_search_results.csv")
