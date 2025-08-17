import csv
import json
from collections import defaultdict

csv_path = './Data_Entry_2017_v2020.csv'
train_json_path = './latex_ocr_train_full_datasets.json'
val_json_path = './latex_ocr_val_full_datasets.json'

# 原始分布统计（你提供的）
original_distribution = {
    "atelectasis": 11559,
    "cardiomegaly": 2776,
    "consolidation": 4667,
    "edema": 2303,
    "effusion": 13317,
    "emphysema": 2516,
    "fibrosis": 1686,
    "hernia": 227,
    "infiltration": 19894,
    "mass": 5782,
    "no finding": 60361,
    "nodule": 6331,
    "pleural thickening": 3385,
    "pneumonia": 1431,
    "pneumothorax": 5302
}

TOTAL_TARGET = 30000

# 计算目标数量（保留整数）
target_distribution = {
    disease: int(count / sum(original_distribution.values()) * TOTAL_TARGET)
    for disease, count in original_distribution.items()
}

# 跟踪每种病已采样数
current_counts = defaultdict(int)

# 清理标签名
def clean_label(label):
    return label.replace("_", " ").lower()

# 构造回答
def build_answer(disease_str):
    disease_list = disease_str.split("|")
    disease_list = [clean_label(d) for d in disease_list]
    if len(disease_list) == 1:
        return f"This chest X-ray image shows {disease_list[0]}."
    else:
        return "This chest X-ray image shows " + ", ".join(disease_list[:-1]) + f", and {disease_list[-1]}."

# 主逻辑
conversations = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        finding = row['Finding Labels']
        image_path = f"full_datasets/{row['Image Index']}"
        disease_list = [clean_label(d) for d in finding.split("|")]

        # 需要跳过含多个标签且有任何一个标签已超额的样本
        if any(current_counts[d] >= target_distribution[d] for d in disease_list if d in target_distribution):
            continue

        # 标记采样
        for d in disease_list:
            if d in target_distribution:
                current_counts[d] += 1

        # 构造对话
        answer = build_answer(finding)
        conversations.append({
            "id": f"identity_{len(conversations)+1}",
            "conversations": [
                {"role": "user", "value": image_path},
                {"role": "assistant", "value": answer}
            ]
        })

        # 终止条件：采够30000个样本
        if len(conversations) >= TOTAL_TARGET:
            break

# 分割训练集和验证集
split_ratio = 0.8
split_index = int(len(conversations) * split_ratio)
train_conversations = conversations[:split_index]
val_conversations = conversations[split_index:]

# 保存文件
with open(train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_conversations, f, ensure_ascii=False, indent=2)

with open(val_json_path, 'w', encoding='utf-8') as f:
    json.dump(val_conversations, f, ensure_ascii=False, indent=2)

print(f"✅ 已生成训练数据 {len(train_conversations)} 条，验证数据 {len(val_conversations)} 条")
print("🎯 每类疾病采样统计：")
for disease in sorted(target_distribution):
    print(f"{disease:<20} : {current_counts[disease]}/{target_distribution[disease]}")
