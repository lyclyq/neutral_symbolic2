import pandas as pd
import json
import os

os.chdir(os.path.dirname(__file__))
# === 路径定义 ===
csv_path = "Data_Entry_2017_v2020.csv"
output_json_path = "all_diseases_full_datasets.json"

# === 读取 CSV ===
df = pd.read_csv(csv_path)

# === 提取、去重 ===
disease_set = set()
for label_str in df["Finding Labels"].dropna():
    labels = [x.strip() for x in label_str.split("|")]
    disease_set.update(labels)

# === 保存 JSON ===
disease_list = sorted(list(disease_set))  # 可排序输出
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(disease_list, f, indent=2)

print(f"✅ 共提取疾病标签数量: {len(disease_list)}，已保存至 {output_json_path}")
