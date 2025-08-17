import torch
import numpy as np
from datasets import Dataset, Features, Value, LargeList
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json
import os
import inspect
import threading
from torch.nn import CrossEntropyLoss
from predict_with_weighted_TP_FP_FN_TN import evaluate
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(compute_dtype)
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
import os
print("Current working directory:", os.getcwd())

# print(inspect.getsource(Trainer.compute_loss))

prompt = "Please describe the image in detail, using no more than 50 words."
model_id = "Qwen/Qwen2-VL-7B-Instruct"
local_model_path = "./Qwen/Qwen2-VL-7B-Instruct"
output_dir = "./output/Qwen2-VL-7B-Instruct-output_2poch_coco"
train_dataset_json_path = "coco/coco_train_conversation.json"
val_dataset_json_path = "coco/coco_val_conversation.json"
# output_dir = "./output/Qwen2-VL-2B-LatexOCR_weighted_2_350_bias_ep3_large_samples"
MAX_LENGTH = 8192









# disease_weights = {disease: 5.0 for disease in disease_list}  # 可调整放大倍率

tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(local_model_path)






bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # 或 "fp4"，建议 nf4 精度更高
    bnb_4bit_compute_dtype=torch.bfloat16  # 若不支持 bfloat16 可换成 torch.float16
)
print("Current working directory:", os.getcwd())
# ✅ 加载模型

# disease_token_map = {}
# max_disease_token_len = 0
# for disease in disease_list:
#     token_ids = tokenizer(disease, add_special_tokens=False)["input_ids"]
#     disease_token_map[disease] = token_ids
#     max_disease_token_len = max(max_disease_token_len, len(token_ids))

# 定义一个超时控制器
def finish_with_timeout(timeout=15):
    def finish_task():
        try:
            swanlab.finish()
        except Exception as e:
            print(f"[Warning] swanlab.finish() failed: {e}")

    thread = threading.Thread(target=finish_task)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        print(f"[Timeout] Swanlab upload took too long (> {timeout}s), exiting upload...")
        # 你可以选择强制退出线程（不推荐），或者就留着它后台上传（推荐）
    else:
        print("✅ Swanlab finish completed within timeout.")

# 执行




def process_func(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    pixel_values_list = []
    image_grid_thw_list = []

    input_ids, attention_mask, labels = [], [], []
    conversation = examples["conversations"]
    image_file_path = conversation[0]["value"]
    output_content = conversation[1]["value"]

    # with Image.open(image_file_path) as img:
    #     width, height = img.size

    # # 调整大小（如果任何一边 > 400）
    # if width > 400 or height > 400:
    #     resized_width = width // 2
    #     resized_height = height // 2
    # else:
    #     resized_width = width
    #     resized_height = height
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_file_path}",
                    "resized_height": 300,
                    "resized_width": 300,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)


    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #由（1,h,w)变换为（h,w）
    return {
        "id": examples["id"],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs['pixel_values'],
        "image_grid_thw": inputs['image_grid_thw']
    }




def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
#     generated_ids = model.generate(
#     **inputs,
#     max_new_tokens=128,  # 足够覆盖多个病症 + 分隔符
#     repetition_penalty=1.3,  # 惩罚重复词
#     eos_token_id=tokenizer.eos_token_id,  # 明确结束符
#     do_sample=False  # greedy 或 beam search 更稳
# )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


# 在modelscope上下载Qwen2-VL模型到本地目录下
# model_dir = snapshot_download(model_id, cache_dir="./", revision="master")

# origin_model = Qwen2VLForConditionalGeneration.from_pretrained(local_model_path, device_map="auto",quantization_config=bnb_config, trust_remote_code=True,)

origin_model = Qwen2VLForConditionalGeneration.from_pretrained(local_model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True,)
origin_model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取json文件
train_ds = Dataset.from_json(train_dataset_json_path)
# train_ds = train_ds.select(range(5))  # 只取前5个样本
# train_dataset = train_ds.map(process_func)
# train_dataset = train_ds.map(process_func, batched=False, batch_size=1, keep_in_memory=True)
# train_ds = load_dataset("json", data_files=train_dataset_json_path, split="train", streaming=False)
# features = Features({
#     "id": Value("string"),
#     "input_ids": LargeList(Value("int32")),
#     "attention_mask": LargeList(Value("int32")),
#     "labels": LargeList(Value("int32")),
#     "pixel_values": LargeList(LargeList(Value("float32"))),  # 2D
#     "image_grid_thw": LargeList(Value("int64")),             # 1D
# })

# # 然后才 map，不需要 keep_in_memory 参数
# train_dataset = train_ds.map(
#     process_func,
#     batched=False,
#     batch_size=1,
#     # features=features,
#     remove_columns=["conversations"],
#     keep_in_memory=True
# )
# 这一步告诉 datasets 用 PyTorch Tensor 表达形式，而不落盘：
# train_dataset.set_format(type="torch")

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["gate_proj", "up_proj", "down_proj"],

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False,  # 训练模式
    # inference_mode = True,  # 训练模式
    r=12,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例 
    bias = "lora_only"
)

# 获取LoRA模型
train_peft_model = get_peft_model(origin_model, config)

# train_peft_model.train()
for name, param in train_peft_model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")
# 配置训练参数
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,                 # 改为 5 epoch
    learning_rate=5e-5,                 # 可选，稍微小一点更稳
    logging_steps=5,                    # 每 step 打印 loss，便于观察
    save_steps=300,                     # 每 100 step 保存一次 checkpoint
    save_total_limit=3,                # 最多保留 2 个模型
    gradient_checkpointing=True,
    save_on_each_node=True,
    report_to="none"
)
        
# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen2-VL-7B-Instruct-full-datasets-weighted-punish",
    experiment_name= output_dir.split("/")[-1],
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct",
        "dataset": "https://modelscope.cn/datasets/AI-ModelScope/LaTeX_OCR/summary",
        # "github": "https://github.com/datawhalechina/self-llm",
        "model_id": model_id,
        "train_dataset_json_path": train_dataset_json_path,
        "val_dataset_json_path": val_dataset_json_path,
        "output_dir": output_dir,
        "prompt": prompt,
        "train_data_number": len(train_ds),
        "token_max_length": MAX_LENGTH,
        "lora_rank": 12,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

# # 配置Trainer
# trainer = Trainer(
#     model=train_peft_model,
#     args=args,
#     train_dataset=train_dataset,
#     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
#     callbacks=[swanlab_callback],
# )

# # 开启模型训练
# trainer.train()


train_dataset = train_ds.map(process_func)
train_dataset.set_format(type="torch")
# train_dataset = train_ds.map(process_func, batched=False, batch_size=1, keep_in_memory=True)
# train_ds = load_dataset("json", data_files=train_dataset_json_path, split="train", streaming=False)
# features = Features({
#     "id": Value("string"),
#     "input_ids": LargeList(Value("int32")),
#     "attention_mask": LargeList(Value("int32")),
#     "labels": LargeList(Value("int32")),
#     "pixel_values": LargeList(LargeList(Value("float32"))),  # 2D
#     "image_grid_thw": LargeList(Value("int64")),             # 1D
# })

# # 然后才 map，不需要 keep_in_memory 参数
# train_dataset = train_ds.map(
#     process_func,
#     batched=False,
#     batch_size=1,
#     # features=features,
#     remove_columns=["conversations"],
#     keep_in_memory=True
# )
# 这一步告诉 datasets 用 PyTorch Tensor 表达形式，而不落盘：


# train_dataset = train_ds.map(process_func, batched=True, batch_size=32)
# train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# 替换训练使用的 Trainer
# trainer = CustomTrainer(
#     model=train_peft_model,
#     args=args,
#     train_dataset=train_dataset,
#     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
#     callbacks=[swanlab_callback],
# )
trainer = Trainer(
    model=train_peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)
# 训练启动
trainer.train()

# ====================测试===================
# model = load_model_with_lora(base_model_path, lora_path)
# tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
# processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
# evaluate(val_dataset_json_path, model, tokenizer, processor, output_dir)
# 配置测试参数
# val_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=True,  # 训练模式
#     r=12,  # Lora 秩
#     lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
#     lora_dropout=0.05,  # Dropout 比例
#     # bias="none",
#     bias = "lora_only"
# )

# # 获取测试模型，从output_dir中获取最新的checkpoint
# checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
# if not checkpoints:
#     raise ValueError(f"在 {output_dir} 下没有找到任何 checkpoint，请先完成训练！")
# load_model_path = f"{output_dir}/checkpoint-{max([int(d.split('-')[-1]) for d in checkpoints])}"
# print(f"load_model_path: {load_model_path}")
# val_peft_model = PeftModel.from_pretrained(origin_model, model_id=load_model_path, config=val_config)

# # 读取测试数据
# with open(val_dataset_json_path, "r") as f:
#     test_dataset = json.load(f)

# # test_dataset = test_dataset[:5]  # 只取前5个样本

# test_image_list = []
# for item in test_dataset:
#     image_file_path = item["conversations"][0]["value"]
#     label = item["conversations"][1]["value"]
    
#     messages = [{
#         "role": "user", 
#         "content": [
#             {
#             "type": "image", 
#             "image": image_file_path,
#             "resized_height": 800,
#             "resized_width": 800,   
#             },
#             {
#             "type": "text",
#             "text": prompt,
#             }
#         ]}]
    
#     response = predict(messages, val_peft_model)
    
#     print(f"predict:{response}")
#     print(f"gt:{label}\n")

#     test_image_list.append(swanlab.Image(image_file_path, caption=response))

# # swanlab.log({"Prediction": test_image_list})
# swanlab.log({"Prediction": test_image_list[:5]})
# # 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
# # swanlab.finish()
# finish_with_timeout(timeout=15)