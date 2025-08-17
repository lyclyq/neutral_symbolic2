import csv
import json

csv_path = './sample_labels.csv'
train_json_path = './latex_ocr_train.json'
val_json_path = './latex_ocr_val.json'

# 你定义的疾病列表（顺序必须固定）
disease_list = [
    "atelectasis", "cardiomegaly", "consolidation", "edema", "effusion",
    "emphysema", "fibrosis", "hernia", "infiltration", "mass", "no finding",
    "nodule", "pleural thickening", "pneumonia", "pneumothorax"
]

# 转换单条标签为 one-hot
def build_onehot(label_str):
    raw_labels = label_str.strip().split("|")
    clean_label = lambda s: s.replace("_", " ").lower()
    labels = [clean_label(l) for l in raw_labels if l.strip()]
    onehot = [1 if disease in labels else 0 for disease in disease_list]
    return onehot, labels

# 主处理流程
dataset = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        label_str = row['Finding Labels'].strip()

        # 跳过空标签（不是 no finding，是完全为空）
        if label_str == "":
            continue

        onehot, disease_names = build_onehot(label_str)

        dataset.append({
            "id": f"identity_{idx+1}",
            "image_path": f"sample/images/{row['Image Index']}",
            "disease_labels": onehot,
            "disease_names": disease_names
        })

# 划分训练集/验证集（按 80/20）
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)
train_data = dataset[:split_index]
val_data = dataset[split_index:]

# 写入 JSON 文件
with open(train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(val_json_path, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"✅ 清洗后生成 JSON：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条。")
