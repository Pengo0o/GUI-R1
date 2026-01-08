# GUI-Agent Training Framework

<div align="center">

**基于 VERL 和 LLaMA-Factory 的 GUI 智能体训练框架**

[English](#english) | [中文](#中文)

</div>

---

## 中文

### 📖 项目简介

本项目是一个综合性的 GUI Agent 训练框架，支持使用多种训练方法和框架来训练视觉-语言模型以执行 GUI 操作任务。项目整合了业界领先的训练框架，包括 VERL、LLaMA-Factory、VLM-R1 和 ms-swift，为 GUI 智能体的训练提供了灵活且强大的解决方案。

### ✨ 主要特性

- 🎯 **多框架支持**：集成 VERL、LLaMA-Factory、VLM-R1、ms-swift
- 🚀 **强化学习训练**：支持 GRPO (Group Relative Policy Optimization)
- 📚 **监督微调**：支持传统 SFT (Supervised Fine-Tuning)
- 🖼️ **多模态能力**：基于 Qwen2.5-VL 等视觉-语言模型
- 🔧 **灵活配置**：支持 LoRA、全参数微调等多种训练方式
- 📊 **完整数据流**：从数据预处理到模型训练的完整工具链

### 🏗️ 项目结构

```
Gui-Agent/
├── Data/                          # 数据处理工具
│   ├── utils/
│   │   ├── convert_to_format.py   # 数据格式转换
│   │   ├── clean_data.py          # 数据清洗
│   │   └── add_solution_field.py  # 添加解决方案字段
│   └── hfd.sh                     # 数据下载脚本
├── verl/                          # VERL 强化学习框架
│   └── examples/
│       └── grpo_trainer/          # GRPO 训练示例
├── LLaMA-Factory/                 # LLaMA-Factory SFT 框架
│   ├── data/
│   │   └── gui-r1-3k.json        # GUI 训练数据
│   └── examples/
│       └── train_lora/            # LoRA 训练示例
├── VLM-R1/                        # VLM-R1 多模态训练框架
│   ├── run_scripts/               # 训练脚本
│   └── src/open-r1-multimodal/    # 核心代码
├── ms-swift/                      # MS-SWIFT 训练框架
└── 1/                             # VERL GUI 训练实验
    ├── examples/
    │   ├── qwen2_5_vl_7b_gui_grpo.sh
    │   └── baselines/
    └── guir1/
        ├── inference.sh
        └── eval.sh
```

### 🚀 快速开始

#### 环境配置

1. **创建 Conda 环境**

```bash
conda create -n gui-agent python=3.10
conda activate gui-agent
```

2. **安装依赖（根据选择的框架）**

**使用 VERL 框架：**
```bash
cd verl
pip install -r requirements.txt
```

**使用 LLaMA-Factory 框架：**
```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

**使用 VLM-R1 框架：**
```bash
cd VLM-R1
bash setup.sh
```

**使用 ms-swift 框架：**
```bash
cd ms-swift
pip install -r requirements.txt
```

### 📊 数据准备

#### 数据格式转换

本项目提供了完整的数据处理工具链，支持将 GUI-R1-3k 数据集转换为各框架所需格式：

```bash
cd Data/utils

# 转换为标准格式
python convert_to_format.py

# 转换为 Swift 格式
python convert_to_swift_format.py

# 数据清洗
python clean_data.py

# 添加解决方案字段
python add_solution_field.py
```

#### 数据格式说明

**VERL 格式（Parquet）：**
```python
{
    'image': str,           # 图像路径
    'instruction': str,     # 任务指令
    'history': str,         # 历史操作
    'gt_action': str,       # 正确动作
    'gt_bbox': list,        # 目标位置
    'gt_input_text': str,   # 输入文本
    'task_type': str        # 任务类型 (high/low)
}
```

**LLaMA-Factory 格式（JSON）：**
```json
{
    "messages": [
        {
            "content": "<image>执行命令...",
            "role": "user"
        },
        {
            "role": "assistant",
            "content": "[{'action': 'click', 'point': [x, y], 'input_text': '...'}]"
        }
    ],
    "images": ["path/to/image.png"]
}
```

### 🎯 训练方法

#### 方法 1: 使用 VERL 进行 GRPO 训练

GRPO (Group Relative Policy Optimization) 是一种强化学习方法，通过奖励模型优化策略。

```bash
cd 1

# 编辑脚本配置
# 修改 MODEL_PATH 为你的模型路径
# 修改数据路径

bash examples/qwen2_5_vl_7b_gui_grpo.sh
```

**关键参数：**
- `model_path`: 基础模型路径（如 Qwen2.5-VL-7B-Instruct）
- `data.train_files`: 训练数据文件
- `worker.reward.compute_score`: 奖励计算方式（r1gui）
- `data.max_pixels`: 图像最大像素数
- `trainer.n_gpus_per_node`: 每节点 GPU 数量

#### 方法 2: 使用 LLaMA-Factory 进行 SFT 训练

SFT (Supervised Fine-Tuning) 适用于有标注数据的监督学习场景。

```bash
cd LLaMA-Factory

# LoRA 微调
llamafactory-cli train examples/train_lora/qwen2_5_vl_3b_gui_lora_sft.yaml

# 全参数微调
llamafactory-cli train examples/train_full/qwen2_5_vl_full_sft.yaml
```

**配置说明（qwen2_5_vl_3b_gui_lora_sft.yaml）：**
```yaml
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
stage: sft
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
dataset: gui-r1-3k
learning_rate: 1.0e-6
num_train_epochs: 1.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
```

#### 方法 3: 使用 VLM-R1 进行 GRPO 训练

VLM-R1 是专门为视觉-语言模型设计的 R1 风格训练框架。

```bash
cd VLM-R1

# 修改 run_scripts/run_grpo_gui.sh 中的路径配置
# 设置 model_path, data_paths, image_folders

bash run_scripts/run_grpo_gui.sh
```

**特点：**
- 支持多图像输入
- 自定义奖励函数
- 支持 LoRA 和全参数训练
- 支持多节点训练

#### 方法 4: 使用 ms-swift 进行训练

ms-swift 是 ModelScope 提供的模型训练框架。

```bash
cd ms-swift

# 查看可用的训练示例
ls examples/train_lora/
ls examples/train_full/

# 运行训练
swift sft --model-type qwen2-vl-7b-instruct \
    --dataset gui-r1-3k \
    --output-dir output/gui-agent
```

### 📈 模型推理与评估

#### 使用 VERL 推理

```bash
cd 1/guir1
bash inference.sh
```

#### 使用 VERL 评估

```bash
cd 1/guir1
bash eval.sh
```

#### 使用 LLaMA-Factory 推理

```bash
cd LLaMA-Factory

# CLI 交互式推理
llamafactory-cli chat \
    --model_name_or_path path/to/checkpoint \
    --template qwen2_vl

# API 部署
llamafactory-cli api \
    --model_name_or_path path/to/checkpoint \
    --template qwen2_vl
```

#### 使用 VLM-R1 评估

```bash
cd VLM-R1/src/eval

# 评估 R1 模型
torchrun --nproc_per_node=8 test_rec_r1.py

# 评估基线模型
torchrun --nproc_per_node=8 test_rec_baseline.py
```

### 🔬 训练框架对比

| 框架 | 训练方法 | 优势 | 适用场景 |
|------|---------|------|---------|
| **VERL** | GRPO (强化学习) | • 支持奖励模型优化<br>• 更好的泛化能力<br>• 适合探索性任务 | 需要优化策略的复杂 GUI 任务 |
| **LLaMA-Factory** | SFT (监督学习) | • 训练稳定<br>• 配置简单<br>• 支持多种微调方式 | 有明确标注数据的场景 |
| **VLM-R1** | GRPO (强化学习) | • 专为视觉-语言模型设计<br>• 支持多图像输入<br>• 丰富的奖励函数 | 多模态推理和视觉理解任务 |
| **ms-swift** | SFT/LoRA | • 集成 ModelScope<br>• 开箱即用<br>• 社区支持好 | 快速原型开发和实验 |

### 💡 最佳实践

#### 选择训练方法

1. **有充足标注数据** → 使用 LLaMA-Factory SFT
2. **需要优化策略** → 使用 VERL 或 VLM-R1 GRPO
3. **快速实验** → 使用 ms-swift
4. **多模态推理** → 使用 VLM-R1

#### 超参数建议

**LoRA 训练：**
```yaml
lora_rank: 8-16
lora_alpha: 16-32 (通常为 rank 的 2 倍)
learning_rate: 1e-4 ~ 5e-4
batch_size: 4-8 per device
```

**全参数训练：**
```yaml
learning_rate: 1e-5 ~ 1e-6
batch_size: 1-2 per device
gradient_accumulation_steps: 4-8
```

**GRPO 训练：**
```yaml
beta: 0.01 ~ 0.04 (KL 散度权重)
num_generations: 4-8 (每步生成样本数)
max_completion_length: 1024-2048
```

### 🛠️ 常见问题

<details>
<summary><b>Q: CUDA 内存不足怎么办？</b></summary>

**解决方案：**
1. 减少 `per_device_train_batch_size`
2. 增加 `gradient_accumulation_steps`
3. 使用 LoRA 而非全参数训练
4. 启用梯度检查点：`gradient_checkpointing: true`
5. 使用 DeepSpeed ZeRO-3 配置
</details>

<details>
<summary><b>Q: 如何选择基础模型？</b></summary>

**推荐模型：**
- **Qwen2.5-VL-3B-Instruct**: 适合资源受限场景，训练快速
- **Qwen2.5-VL-7B-Instruct**: 平衡性能和资源，推荐使用
- **Qwen2.5-VL-72B-Instruct**: 最佳性能，需要大量 GPU
</details>

<details>
<summary><b>Q: 数据格式转换失败？</b></summary>

**检查清单：**
1. 确保图像路径正确
2. 检查数据字段完整性
3. 验证 JSON/Parquet 格式是否正确
4. 查看转换脚本的错误日志
</details>

<details>
<summary><b>Q: 训练不收敛？</b></summary>

**调试步骤：**
1. 检查学习率是否过大
2. 验证数据质量
3. 尝试更小的 beta 值（GRPO）
4. 检查奖励函数是否合理
5. 查看 wandb/tensorboard 训练曲线
</details>

### 📚 相关资源

- [GUI-R1](https://github.com/ritzz-ai/GUI-R1)
- [VERL 文档](./verl/README.md)
- [LLaMA-Factory 文档](./LLaMA-Factory/README.md)
- [VLM-R1 文档](./VLM-R1/README.md)
- [ms-swift 文档](./ms-swift/README.md)
- [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)

### 🤝 致谢

本项目整合了以下开源项目：

- [GUI-R1](https://github.com/ritzz-ai/GUI-R1) - GUIR1模型
- [VERL](https://github.com/volcengine/verl) - 强化学习训练框架
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 大模型微调工具
- [VLM-R1](https://github.com/om-ai-lab/VLM-R1) - 视觉-语言模型 R1 训练
- [ms-swift](https://github.com/modelscope/swift) - ModelScope 训练框架
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) - 视觉-语言基础模型

### 📝 许可证

本项目遵循各子项目的原始许可证。

---

## English

### 📖 Project Overview

This is a comprehensive GUI Agent training framework that supports multiple training methods and frameworks for training vision-language models to perform GUI operation tasks. The project integrates industry-leading training frameworks including VERL, LLaMA-Factory, VLM-R1, and ms-swift, providing a flexible and powerful solution for GUI agent training.

### ✨ Key Features

- 🎯 **Multi-Framework Support**: Integrates VERL, LLaMA-Factory, VLM-R1, and ms-swift
- 🚀 **Reinforcement Learning**: Supports GRPO (Group Relative Policy Optimization)
- 📚 **Supervised Fine-tuning**: Supports traditional SFT methods
- 🖼️ **Multimodal Capabilities**: Based on vision-language models like Qwen2.5-VL
- 🔧 **Flexible Configuration**: Supports LoRA, full fine-tuning, and more
- 📊 **Complete Pipeline**: Full toolchain from data preprocessing to model training

### 🏗️ Project Structure

```
Gui-Agent/
├── Data/                          # Data processing tools
│   ├── utils/
│   │   ├── convert_to_format.py   # Data format conversion
│   │   ├── clean_data.py          # Data cleaning
│   │   └── add_solution_field.py  # Add solution field
│   └── hfd.sh                     # Data download script
├── verl/                          # VERL reinforcement learning framework
│   └── examples/
│       └── grpo_trainer/          # GRPO training examples
├── LLaMA-Factory/                 # LLaMA-Factory SFT framework
│   ├── data/
│   │   └── gui-r1-3k.json        # GUI training data
│   └── examples/
│       └── train_lora/            # LoRA training examples
├── VLM-R1/                        # VLM-R1 multimodal training framework
│   ├── run_scripts/               # Training scripts
│   └── src/open-r1-multimodal/    # Core code
├── ms-swift/                      # MS-SWIFT training framework
└── 1/                             # VERL GUI training experiments
    ├── examples/
    │   ├── qwen2_5_vl_7b_gui_grpo.sh
    │   └── baselines/
    └── guir1/
        ├── inference.sh
        └── eval.sh
```

### 🚀 Quick Start

#### Environment Setup

1. **Create Conda Environment**

```bash
conda create -n gui-agent python=3.10
conda activate gui-agent
```

2. **Install Dependencies (Choose Your Framework)**

**For VERL:**
```bash
cd verl
pip install -r requirements.txt
```

**For LLaMA-Factory:**
```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

**For VLM-R1:**
```bash
cd VLM-R1
bash setup.sh
```

**For ms-swift:**
```bash
cd ms-swift
pip install -r requirements.txt
```

### 📊 Data Preparation

#### Data Format Conversion

The project provides a complete data processing toolchain to convert GUI-R1-3k dataset to required formats:

```bash
cd Data/utils

# Convert to standard format
python convert_to_format.py

# Convert to Swift format
python convert_to_swift_format.py

# Data cleaning
python clean_data.py

# Add solution field
python add_solution_field.py
```

#### Data Format Specification

**VERL Format (Parquet):**
```python
{
    'image': str,           # Image path
    'instruction': str,     # Task instruction
    'history': str,         # Action history
    'gt_action': str,       # Ground truth action
    'gt_bbox': list,        # Target location
    'gt_input_text': str,   # Input text
    'task_type': str        # Task type (high/low)
}
```

**LLaMA-Factory Format (JSON):**
```json
{
    "messages": [
        {
            "content": "<image>Execute command...",
            "role": "user"
        },
        {
            "role": "assistant",
            "content": "[{'action': 'click', 'point': [x, y], 'input_text': '...'}]"
        }
    ],
    "images": ["path/to/image.png"]
}
```

### 🎯 Training Methods

#### Method 1: GRPO Training with VERL

GRPO (Group Relative Policy Optimization) is a reinforcement learning method that optimizes policy through reward models.

```bash
cd 1

# Edit script configuration
# Modify MODEL_PATH to your model path
# Modify data paths

bash examples/qwen2_5_vl_7b_gui_grpo.sh
```

**Key Parameters:**
- `model_path`: Base model path (e.g., Qwen2.5-VL-7B-Instruct)
- `data.train_files`: Training data file
- `worker.reward.compute_score`: Reward computation method (r1gui)
- `data.max_pixels`: Maximum image pixels
- `trainer.n_gpus_per_node`: Number of GPUs per node

#### Method 2: SFT Training with LLaMA-Factory

SFT (Supervised Fine-Tuning) is suitable for supervised learning scenarios with labeled data.

```bash
cd LLaMA-Factory

# LoRA fine-tuning
llamafactory-cli train examples/train_lora/qwen2_5_vl_3b_gui_lora_sft.yaml

# Full fine-tuning
llamafactory-cli train examples/train_full/qwen2_5_vl_full_sft.yaml
```

**Configuration (qwen2_5_vl_3b_gui_lora_sft.yaml):**
```yaml
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
stage: sft
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
dataset: gui-r1-3k
learning_rate: 1.0e-6
num_train_epochs: 1.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
```

#### Method 3: GRPO Training with VLM-R1

VLM-R1 is a R1-style training framework specifically designed for vision-language models.

```bash
cd VLM-R1

# Modify path configurations in run_scripts/run_grpo_gui.sh
# Set model_path, data_paths, image_folders

bash run_scripts/run_grpo_gui.sh
```

**Features:**
- Multi-image input support
- Customizable reward functions
- Support for LoRA and full parameter training
- Multi-node training support

#### Method 4: Training with ms-swift

ms-swift is a model training framework provided by ModelScope.

```bash
cd ms-swift

# View available training examples
ls examples/train_lora/
ls examples/train_full/

# Run training
swift sft --model-type qwen2-vl-7b-instruct \
    --dataset gui-r1-3k \
    --output-dir output/gui-agent
```

### 📈 Inference and Evaluation

#### Inference with VERL

```bash
cd 1/guir1
bash inference.sh
```

#### Evaluation with VERL

```bash
cd 1/guir1
bash eval.sh
```

#### Inference with LLaMA-Factory

```bash
cd LLaMA-Factory

# CLI interactive inference
llamafactory-cli chat \
    --model_name_or_path path/to/checkpoint \
    --template qwen2_vl

# API deployment
llamafactory-cli api \
    --model_name_or_path path/to/checkpoint \
    --template qwen2_vl
```

#### Evaluation with VLM-R1

```bash
cd VLM-R1/src/eval

# Evaluate R1 model
torchrun --nproc_per_node=8 test_rec_r1.py

# Evaluate baseline
torchrun --nproc_per_node=8 test_rec_baseline.py
```

### 🔬 Framework Comparison

| Framework | Method | Advantages | Use Cases |
|-----------|--------|------------|-----------|
| **VERL** | GRPO (RL) | • Reward model optimization<br>• Better generalization<br>• Good for exploration | Complex GUI tasks requiring policy optimization |
| **LLaMA-Factory** | SFT | • Stable training<br>• Simple configuration<br>• Multiple tuning methods | Scenarios with clear labeled data |
| **VLM-R1** | GRPO (RL) | • Designed for VLMs<br>• Multi-image support<br>• Rich reward functions | Multimodal reasoning and vision tasks |
| **ms-swift** | SFT/LoRA | • ModelScope integration<br>• Easy to use<br>• Good community support | Rapid prototyping and experiments |

### 💡 Best Practices

#### Choosing Training Methods

1. **Sufficient labeled data** → Use LLaMA-Factory SFT
2. **Need policy optimization** → Use VERL or VLM-R1 GRPO
3. **Quick experiments** → Use ms-swift
4. **Multimodal reasoning** → Use VLM-R1

#### Hyperparameter Recommendations

**LoRA Training:**
```yaml
lora_rank: 8-16
lora_alpha: 16-32 (usually 2x rank)
learning_rate: 1e-4 ~ 5e-4
batch_size: 4-8 per device
```

**Full Parameter Training:**
```yaml
learning_rate: 1e-5 ~ 1e-6
batch_size: 1-2 per device
gradient_accumulation_steps: 4-8
```

**GRPO Training:**
```yaml
beta: 0.01 ~ 0.04 (KL divergence weight)
num_generations: 4-8 (samples per step)
max_completion_length: 1024-2048
```

### 🛠️ Common Issues

<details>
<summary><b>Q: CUDA out of memory?</b></summary>

**Solutions:**
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Use LoRA instead of full fine-tuning
4. Enable gradient checkpointing: `gradient_checkpointing: true`
5. Use DeepSpeed ZeRO-3 configuration
</details>

<details>
<summary><b>Q: How to choose a base model?</b></summary>

**Recommended Models:**
- **Qwen2.5-VL-3B-Instruct**: For resource-constrained scenarios, fast training
- **Qwen2.5-VL-7B-Instruct**: Balanced performance and resources (recommended)
- **Qwen2.5-VL-72B-Instruct**: Best performance, requires many GPUs
</details>

<details>
<summary><b>Q: Data format conversion failed?</b></summary>

**Checklist:**
1. Ensure image paths are correct
2. Check data field completeness
3. Verify JSON/Parquet format correctness
4. Review conversion script error logs
</details>

<details>
<summary><b>Q: Training not converging?</b></summary>

**Debugging Steps:**
1. Check if learning rate is too large
2. Verify data quality
3. Try smaller beta value (GRPO)
4. Check if reward function is reasonable
5. Review wandb/tensorboard training curves
</details>

### 📚 Related Resources

- [GUI-R1](https://github.com/ritzz-ai/GUI-R1)
- [VERL Documentation](./verl/README.md)
- [LLaMA-Factory Documentation](./LLaMA-Factory/README.md)
- [VLM-R1 Documentation](./VLM-R1/README.md)
- [ms-swift Documentation](./ms-swift/README.md)
- [Qwen3-VL Official Repository](https://github.com/QwenLM/Qwen3-VL)

### 🤝 Acknowledgements

This project integrates the following open-source projects:

- [GUI-R1](https://github.com/ritzz-ai/GUI-R1) - GUI-R1
- [VERL](https://github.com/volcengine/verl) - Reinforcement learning framework
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - LLM fine-tuning tool
- [VLM-R1](https://github.com/om-ai-lab/VLM-R1) - Vision-language model R1 training
- [ms-swift](https://github.com/modelscope/swift) - ModelScope training framework
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) - Vision-language foundation model

### 📝 License

This project follows the original licenses of each sub-project.

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

</div>