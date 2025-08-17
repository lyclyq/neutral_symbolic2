import json
import os
from collections import defaultdict

# 路径配置
json_path_train = './coco/captions_train2017.json'
json_path_val = './coco/captions_val2017.json'
train_json_out = './coco/coco_train_conversation.json'
val_json_out = './coco/coco_val_conversation.json'
TOTAL_TARGET = 30000
def build_conversations(json_path, image_root):
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    image_id_to_file = {
        img["id"]: img["file_name"] for img in coco_data["images"]
    }

    image_id_to_captions = defaultdict(list)
    for ann in coco_data["annotations"]:
        image_id_to_captions[ann["image_id"]].append(ann["caption"])

    conversations = []
    for image_id in image_id_to_captions:
        if image_id not in image_id_to_file:
            continue

        image_path = f"coco/{image_root}/{image_id_to_file[image_id]}"
        if not os.path.exists(image_path):
            continue  # 跳过找不到图片的样本

        captions = image_id_to_captions[image_id]
        if not captions:
            continue

        answer = captions[0]
        conversations.append({
            "id": f"identity_{len(conversations) + 1}",
            "conversations": [
                {"role": "user", "value": image_path},
                {"role": "assistant", "value": answer}
            ]
        })

        if len(conversations) >= TOTAL_TARGET:
            break

    return conversations


# 构造训练集
train_data = build_conversations(json_path_train, image_root="train2017")
val_data = build_conversations(json_path_val, image_root="val2017")

# 保存为 json 文件
os.makedirs(os.path.dirname(train_json_out), exist_ok=True)
with open(train_json_out, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
with open(val_json_out, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"✅ 数据构建完成：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条")
