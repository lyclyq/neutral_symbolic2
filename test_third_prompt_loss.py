import torch
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration

# 模型路径
model_path = "./Qwen/Qwen2-VL-7B-Instruct"

# 加载模型与处理器
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path)

# 假设的多标签预测结果
predicted_diseases = ["cardiomegaly", "edema", "effusion"]

# 构造用户 prompt（不包含图像）
user_prompt = (
    f"The following abnormalities were found in the X-ray image: {', '.join(predicted_diseases)}. "
    "Please write a diagnosis sentence describing the X-ray findings clearly."
)

# 构造对话（无图）
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_prompt
            }
        ]
    }
]

# 预处理成模型输入（不含 image/video）
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to(model.device)

# 推理
model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# 输出结果
print("Model Diagnosis Output:\n", result)
