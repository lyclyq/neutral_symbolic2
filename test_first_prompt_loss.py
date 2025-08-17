import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration

model_path = "./Qwen/Qwen2-VL-7B-Instruct"

# 加载原始模型（未微调）
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path)

#  系统提示（System Prompt）
system_prompt = (
    "Is this image a chest X-ray? Is the user asking about abnormalities in the image? "
    "Please answer with 'yes, yes', 'yes, no', 'no, yes', or 'no, no'. "
    "Do not explain. Only answer like: 'yes, yes'."
)

#  用户实际问题 user_prompt
user_prompt = "What abnormalities are shown in this chest X-ray?"

# 输入图片路径 image_path
image_path = "./sample/images/00000013_005.png"  # 替换为你的图像路径

# 构造多模态消息 messages
messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
                "resized_height": 400,
                "resized_width": 400
            },
            {
                "type": "text",
                "text": user_prompt
            }
        ]
    }
]

#  预处理成模型输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt"
).to(model.device)

#  推理
model.eval()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(" Model Response:", response)
