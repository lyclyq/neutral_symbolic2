import json
import csv

# ✅ 替换为你本地的实际路径
json_path = "./coco/captions_train2017.json"
csv_path = "./coco/captions_train2017.csv"

# 读取 JSON 文件
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取 annotations 中的 image_id 和 caption
annotations = data["annotations"]

# 写入 CSV
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_id", "caption"])  # 表头
    for ann in annotations:
        writer.writerow([ann["image_id"], ann["caption"]])

print(f"✅ 提取完成，共保存 {len(annotations)} 条 caption 到 {csv_path}")
