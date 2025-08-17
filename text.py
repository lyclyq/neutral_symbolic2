from datasets import load_dataset

# 加载完整的 chest x-ray 数据集
dataset = load_dataset("sbhatti/ChestX-ray14", split="train")

# 看一个样例
print(dataset[0])