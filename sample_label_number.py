import pandas as pd
from collections import Counter

# === 配置 ===
csv_path = "Data_Entry_2017_v2020.csv"  # 替换为你的实际路径
disease_list = [
    "atelectasis", "cardiomegaly", "consolidation", "edema", "effusion",
    "emphysema", "fibrosis", "hernia", "infiltration", "mass", "no finding",
    "nodule", "pleural thickening", "pneumonia", "pneumothorax"
]

# === 加载 CSV ===
df = pd.read_csv(csv_path)

# === 预处理标签 ===
# 把标签统一为小写，去掉空格，处理分隔符
all_labels = []
for labels in df["Finding Labels"]:
    label_items = [l.strip().lower().replace("_", " ") for l in labels.split("|")]
    all_labels.extend(label_items)

# === 统计 ===
counter = Counter(all_labels)

# === 输出统计结果 ===
for disease in disease_list:
    print(f"{disease:20s}: {counter[disease]}")

# === 可选：保存为 CSV ===
output_df = pd.DataFrame([
    {"disease": disease, "count": counter[disease]}
    for disease in disease_list
])
output_df.to_csv("disease_distribution_full_datasets.csv", index=False)
