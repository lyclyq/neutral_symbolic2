<<<<<<< HEAD
# neutral_symbolic
neutral_symbolic
=======
# Qwen2-VL Medical Image Analysis Project

基于 Qwen2-VL-7B-Instruct 的医学影像分析项目，专注于胸部X光片的疾病识别和分类。

## 项目概述

这个项目使用 Qwen2-VL 多模态大语言模型来分析胸部X光片，能够识别和分类多种肺部疾病。项目采用了LoRA (Low-Rank Adaptation) 微调技术来优化模型性能。

## 支持的疾病类型

项目可以识别以下15种疾病：
- atelectasis (肺不张)
- cardiomegaly (心脏肥大)
- consolidation (肺实变)
- edema (肺水肿)
- effusion (胸腔积液)
- emphysema (肺气肿)
- fibrosis (肺纤维化)
- hernia (疝气)
- infiltration (浸润)
- mass (肿块)
- no finding (无异常发现)
- nodule (结节)
- pleural thickening (胸膜增厚)
- pneumonia (肺炎)
- pneumothorax (气胸)

## 项目结构

```
qwen_instruct_model/
├── train_second_part_logical_chain.py  # 主训练脚本
├── train_with_weighted_now_best.py     # 加权训练版本
├── train.py                            # 基础训练脚本
├── inference.py                        # 推理脚本
├── predict_with_weighted_TP_FP_FN_TN.py # 预测和评估
├── latex_ocr_train.json               # 训练数据
├── latex_ocr_val.json                 # 验证数据
├── sample/                            # 样本图片
│   └── images/                        # 胸部X光片样本
├── Qwen/                              # 预训练模型
│   └── Qwen2-VL-7B-Instruct/         # Qwen2-VL-7B模型文件
└── output/                            # 训练输出（不包含在git中）
```

## 安装要求

```bash
pip install torch torchvision transformers
pip install peft datasets
pip install qwen-vl-utils
pip install swanlab
pip install modelscope
```

## 使用方法

### 1. 训练模型

```bash
# 基础训练
python train.py

# 使用加权损失函数训练
python train_with_weighted_now_best.py

# 逻辑链训练（分类头 + LoRA）
python train_second_part_logical_chain.py
```

### 2. 模型推理

```bash
python inference.py
```

### 3. 预测和评估

```bash
python predict_with_weighted_TP_FP_FN_TN.py
```

## 训练配置

- **模型**: Qwen2-VL-7B-Instruct
- **微调方法**: LoRA (r=12, alpha=16, dropout=0.05)
- **批次大小**: 4 (per device)
- **梯度累积步数**: 4
- **学习率**: 5e-5
- **训练轮数**: 2-6 epochs
- **图像分辨率**: 350x350

## 数据格式

训练数据格式 (JSON):
```json
{
  "id": "identity_1",
  "image_path": "sample/images/00000013_005.png",
  "disease_labels": [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
  "disease_names": ["emphysema", "infiltration", "pleural thickening", "pneumothorax"]
}
```

## 特性

- **多模态学习**: 结合图像和文本信息
- **多标签分类**: 支持同时识别多种疾病
- **LoRA微调**: 高效的参数更新策略
- **量化支持**: 支持8bit量化以减少内存使用
- **实验跟踪**: 集成SwanLab进行实验管理

## 模型架构

1. **骨干网络**: Qwen2-VL-7B-Instruct (冻结)
2. **LoRA适配器**: 微调特定层
3. **分类头**: 
   - Linear(3584 → 1024) + ReLU + Dropout(0.1)
   - Linear(1024 → 15)
4. **损失函数**: BCEWithLogitsLoss (多标签分类)

## 输出结果

训练完成后，模型会输出：
- 检查点文件 (checkpoint-xxx/)
- 分类准确率统计
- 混淆矩阵图片
- 样本预测结果

## 注意事项

1. 确保有足够的GPU内存 (建议16GB+)
2. 图像路径需要相对于项目根目录
3. 训练数据需要预先标注好疾病标签
4. output/ 目录包含大型模型文件，不上传到Git

## 许可证

本项目仅供学术研究使用。使用的Qwen2-VL模型请遵循其原始许可证。

## 联系方式

如有问题，请通过Issue或邮件联系。 
>>>>>>> 4dcb1c2 (Initial commit: Qwen2-VL Medical Image Analysis Project)
