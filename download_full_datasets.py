from datasets import load_dataset

dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", trust_remote_code=True)
print(dataset['train'][0]['labels'])
# 选择训练集部分（或 test）
train_dataset = dataset["train"]

# 然后再访问 features
label_names = train_dataset.features["labels"].feature.names
print(label_names)

['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
