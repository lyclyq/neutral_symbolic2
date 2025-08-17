import os
import shutil
from datasets import load_dataset

# 加载数据集（你可以换成 "test"）
dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", trust_remote_code=True, split="train")

# 输出图像目标文件夹
output_dir = "./NIH_full_images"
os.makedirs(output_dir, exist_ok=True)

for i, example in enumerate(dataset):
    image_obj = example["image"]
    image_path = image_obj.filename  # <-- 这是你要的真实路径
    if image_path is None:
        continue  # 某些情况下为 None，跳过
    
    image_name = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(output_dir, image_name))

    if i % 1000 == 0:
        print(f"✅ Copied {i} images...")

print(f"\n🎉 所有图像已复制到 {output_dir}")
