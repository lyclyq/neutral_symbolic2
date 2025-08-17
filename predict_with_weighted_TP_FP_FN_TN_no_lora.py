import os
import json
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
# === 设置 ===
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

prompt = "What abnormalities are shown in this chest X-ray image? Please list all abnormalities you can identify. If there are multiple conditions, list each one clearly. Limit your response to 20 words."
DISEASE_LIST = [
    "atelectasis", "cardiomegaly", "consolidation", "edema", "effusion",
    "emphysema", "fibrosis", "hernia", "infiltration", "mass", "no finding",
    "nodule", "pleural thickening", "pneumonia", "pneumothorax"
]
# 疾病提取函数 
def extract_diseases(text):
    diseases = [d for d in DISEASE_LIST if d in text.lower()]
    # 归一化no finding同义词
    normalized = []
    for d in diseases:
        if d in ["no abnormality", "no findings", "no abnormalities"]:
            normalized.append("no finding")
        else:
            normalized.append(d)
    return list(set(normalized))  # 去重

# 单图预测函数 
def predict(model, processor, image_path, question):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [
    # {
    #     "role": "system",
    #     "content": "You are a medical image assistant. "
    # },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
                "resized_height": 300,
                "resized_width": 300
            },
            {"type": "text", "text": question}
        ]
    }
]


    # 构建 prompt
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 图像预处理
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
#     generated_ids = model.generate(
#     **inputs,
#     max_new_tokens=128,
#     repetition_penalty=1.3,
#     eos_token_id=tokenizer.eos_token_id,
#     num_beams=5,
#     no_repeat_ngram_size=3,        # ✅ 限制重复的 n-gram
#     length_penalty=1.0,            # ✅ 控制输出长度：<1鼓励短句，>1鼓励长句
#     early_stopping=True,           # ✅ 如果所有 beam 都以 EOS 结束就提前结束
# )


    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text


# 主评估函数 
# def evaluate(val_json_path, model, tokenizer, processor, output_dir):
#     with open(val_json_path, 'r') as f:
#         val_data = json.load(f)
#         val_data = val_data[:500]

#     all_true, all_pred, records = [], [], []
#     TP, FP, FN, TN = 0, 0, 0, 0

#     for item in val_data:
#         image_path = item["conversations"][0]["value"]
#         gt_text = item["conversations"][1]["value"]
#         # pred_text = predict(model, processor, image_path, "What abnormaliies are shown in thties are shown in this chest X-ray image?")
#         pred_text = predict(model, processor, image_path, prompt)
#         if "assistant" in pred_text:
#             pred_text = pred_text.split("assistant")[-1].strip()

#         # if "<|end_of_utterance|>" in pred_text:
#         #     pred_text = pred_text.split("<|end_of_utterance|>")[-1].strip()


#                 # 打印编号、图片路径
#         print(f"\n Sample | Image: {image_path}")
        
#         # 打印 Ground Truth
#         print(f"GT Answer   : {gt_text}")
        
#         # 打印 Model Prediction
#         print(f"Predicted   : {pred_text}")

#         gt_labels = extract_diseases(gt_text)
#         pred_labels = extract_diseases(pred_text)
#         print('gt_labels, pred_labels',gt_labels, pred_labels)

#         y_true, y_pred = [], []
#         row = {"image": image_path, "gt_text": gt_text, "pred_text": pred_text}

#         for d in DISEASE_LIST:
#             gt = int(d in gt_labels)
#             pred = int(d in pred_labels)
#             y_true.append(gt)
#             y_pred.append(pred)
#             row[f"gt_{d}"] = gt
#             row[f"pred_{d}"] = pred         
#             row[f"gt_labels_{d}"] = gt_labels
#             row[f"pred_labels_{d}"] = pred_labels


#             if gt == 1 and pred == 1:
#                 TP += 1
#             elif gt == 0 and pred == 1:
#                 FP += 1
#             elif gt == 1 and pred == 0:
#                 FN += 1
#             elif gt == 0 and pred == 0:
#                 TN += 1

#         records.append(row)
#         all_true.append(y_true)
#         all_pred.append(y_pred)

#     df = pd.DataFrame(records)
#     df.to_csv(os.path.join(output_dir, "per_sample_predictions.csv"), index=False)

#     # 混淆矩阵和热图
#     cm = confusion_matrix(sum(all_true, []), sum(all_pred, []))
#     plt.figure(figsize=(6,5))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"])
#     plt.title("Confusion Matrix")
#     plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
#     plt.close()

#     # 统计 TP, FP, FN, TN

#     precision = TP / (TP + FP + 1e-6)
#     recall = TP / (TP + FN + 1e-6)
#     f1 = 2 * precision * recall / (precision + recall + 1e-6)
#     accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)

#     stats = pd.DataFrame.from_dict({
#     "TP": [TP], "FP": [FP], "FN": [FN], "TN": [TN],
#     "Precision": [precision],
#     "Recall": [recall],
#     "F1": [f1],
#     "Accuracy": [accuracy]
# })

#     print('precision, recall, f1, accuracy', precision, recall, f1, accuracy)
#     stats.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)

def evaluate(val_json_path, model, tokenizer, processor, output_dir):
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
        val_data = val_data[:500]

    all_true, all_pred, records = [], [], []
    TP, FP, FN, TN = 0, 0, 0, 0

    per_class_correct = {d: 0 for d in DISEASE_LIST}
    per_class_total_pred = {d: 0 for d in DISEASE_LIST}

    for item in val_data:
        image_path = item["conversations"][0]["value"]
        gt_text = item["conversations"][1]["value"]
        pred_text = predict(model, processor, image_path, prompt)
        if "assistant" in pred_text:
            pred_text = pred_text.split("assistant")[-1].strip()

        print(f"\n Sample | Image: {image_path}")
        print(f"GT Answer   : {gt_text}")
        print(f"Predicted   : {pred_text}")

        gt_labels = extract_diseases(gt_text)
        pred_labels = extract_diseases(pred_text)
        print('gt_labels, pred_labels', gt_labels, pred_labels)

        y_true, y_pred = [], []
        row = {"image": image_path, "gt_text": gt_text, "pred_text": pred_text}

        for d in DISEASE_LIST:
            gt = int(d in gt_labels)
            pred = int(d in pred_labels)
            y_true.append(gt)
            y_pred.append(pred)
            row[f"gt_{d}"] = gt
            row[f"pred_{d}"] = pred
            row[f"gt_labels_{d}"] = gt_labels
            row[f"pred_labels_{d}"] = pred_labels

            if d in pred_labels:
                per_class_total_pred[d] += 1
                if d in gt_labels:
                    per_class_correct[d] += 1

            if gt == 1 and pred == 1:
                TP += 1
            elif gt == 0 and pred == 1:
                FP += 1
            elif gt == 1 and pred == 0:
                FN += 1
            elif gt == 0 and pred == 0:
                TN += 1

        records.append(row)
        all_true.append(y_true)
        all_pred.append(y_pred)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "per_sample_predictions.csv"), index=False)

    # 混淆矩阵和热图
    cm = confusion_matrix(sum(all_true, []), sum(all_pred, []))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # 全局指标
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    stats = pd.DataFrame.from_dict({
        "TP": [TP], "FP": [FP], "FN": [FN], "TN": [TN],
        "Precision": [precision],
        "Recall": [recall],
        "F1": [f1],
        "Accuracy": [accuracy]
    })
    print('stats', stats)
    stats.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)

    # ✅ 每类准确率统计
    per_disease_stats = {
        "disease": [],
        "correct": [],
        "total_pred": [],
        "per_class_precision": []
    }
    print('per_class_correct', per_class_correct)
    for d in DISEASE_LIST:
        per_disease_stats["disease"].append(d)
        per_disease_stats["correct"].append(per_class_correct[d])
        per_disease_stats["total_pred"].append(per_class_total_pred[d])
        if per_class_total_pred[d] > 0:
            per_disease_stats["per_class_precision"].append(per_class_correct[d] / per_class_total_pred[d])
        else:
            per_disease_stats["per_class_precision"].append(0.0)

    df_disease = pd.DataFrame(per_disease_stats)
    df_disease.to_csv(os.path.join(output_dir, "per_disease_precision.csv"), index=False)


#  加载模型和评估 
def load_base_model(base_model_path, dtype=torch.float16):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()
    return base_model

if __name__ == "__main__":
    base_model_path = "Qwen/Qwen2-VL-7B-Instruct"
    # val_json_path = "latex_ocr_val.json"
    val_json_path ="D:/Project/qwen_instruct_model/latex_ocr_val.json"
    output_dir = "output/non_lora_eval"
    # 自动创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    model = load_base_model(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    evaluate(val_json_path, model, tokenizer, processor, output_dir)
