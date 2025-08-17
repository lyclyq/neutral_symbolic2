# === Step 2: Multi-label Classification Training with LoRA and Custom Labels ===

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from swanlab.integration.transformers import SwanLabCallback
import swanlab
import os
# === global parameters ===
prompt = "What abnormalities are shown in this chest X-ray image?"
model_id = "Qwen/Qwen2-VL-7B-Instruct"
local_model_path = "./Qwen/Qwen2-VL-7B-Instruct"
train_dataset_json_path = "latex_ocr_train.json"
val_dataset_json_path = "latex_ocr_val.json"
output_dir = "./output/Qwen2-VL-2B-Step2"
MAX_LENGTH = 8192

# === custom model (inherit Qwen2VL + classifier) ===
class QwenWithClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model  # Qwen2VLForConditionalGeneration with LoRA
        hidden_size = base_model.config.hidden_size
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 15)
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        labels=None,
        **kwargs
    ):
        # print("\n [Forward] Entered QwenWithClassifier.forward")
        # if input_ids is not None:
        #     print(f"  input_ids shape: {input_ids.shape}")
        # if attention_mask is not None:
        #     print(f"  attention_mask shape: {attention_mask.shape}")
        # if pixel_values is not None:
        #     print(f"  pixel_values shape: {pixel_values.shape}")
        # if image_grid_thw is not None:
        #     print(f"  image_grid_thw shape: {image_grid_thw.shape}")
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                **kwargs
            )




        hidden_states = outputs.hidden_states[-1]  # [B, T, H]
        # print(f"  hidden_states shape: {hidden_states.shape}")

        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        # print(f"  mask shape: {mask.shape}")

        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # [B, H]
        # print(f"  pooled shape: {pooled.shape}")

        logits = self.classification_head(pooled)  # [B, 15]
        # print(f"  logits shape: {logits.shape}")
        # print(" [Forward] QwenWithClassifier forward finished\n")

        return {"logits": logits}

class ClassifierTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  
        labels = inputs.pop("labels")  # [B, 15]
        outputs = model(**inputs)
        logits = outputs["logits"]  # [B, 15]
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())

        # === print debug information ===
        # print("\n[Debug] Batch Input Summary")
        # print("input_ids:", inputs["input_ids"].shape)
        # print("attention_mask:", inputs["attention_mask"].shape)
        # print("pixel_values:", inputs["pixel_values"].shape)
        # print("image_grid_thw:", inputs["image_grid_thw"].shape)
        # print("logits:", logits.shape)
        # print("labels:", labels.shape)
        # print("loss:", loss.item())

        return (loss, outputs) if return_outputs else loss

# === load model and LoRA and replace the structure ===
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(local_model_path)

# origin_model = Qwen2VLForConditionalGeneration.from_pretrained(
#     local_model_path,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# )
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    local_model_path,
    device_map="auto",
    # torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    trust_remote_code=True,
)
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
model = QwenWithClassifier(base_model)


# === freeze the backbone, only train the classifier and LoRA parameters ===
for name, param in model.named_parameters():
    if ("classification_head" in name) or ("lora" in name):
        param.requires_grad = True
    else:
        param.requires_grad = False
# === preprocess function ===
def process_func_step2(example):
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





# === load and process the dataset ===
train_ds = Dataset.from_json(train_dataset_json_path)

# train_ds = train_ds.select(range(8))
train_dataset = train_ds.map(process_func_step2)
train_dataset.set_format(type="torch")


swanlab_callback = SwanLabCallback(
    project="Qwen2-VL-ft-latexocr",
    experiment_name="7B-1kdata",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct",
        "dataset": "https://modelscope.cn/datasets/AI-ModelScope/LaTeX_OCR/summary",
        # "github": "https://github.com/datawhalechina/self-llm",
        "model_id": model_id,
        "train_dataset_json_path": train_dataset_json_path,
        "val_dataset_json_path": val_dataset_json_path,
        "output_dir": output_dir,
        "prompt": prompt,
        "train_data_number": len(train_ds),
        "token_max_length": MAX_LENGTH,
        "lora_rank": 12,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

# === training parameters configuration ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
)

# === start training ===
trainer = ClassifierTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    callbacks=[swanlab_callback],
)

trainer.train()

# save the classifier + LoRA parameters (not including Qwen backbone) ===
output_classifier_path = f"{output_dir}/classifier_only"
os.makedirs(output_classifier_path, exist_ok=True)

# filter parameters
to_save = {
    k: v.cpu()
    for k, v in model.state_dict().items()
    if ("classification_head" in k or "lora" in k)
}

torch.save(to_save, f"{output_classifier_path}/pytorch_model.bin")

# save tokenizer / processor (optional)
tokenizer.save_pretrained(output_classifier_path)
processor.save_pretrained(output_classifier_path)

print(f"classifier + LoRA parameters saved to: {output_classifier_path}")

