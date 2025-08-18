<div align="center">
<br>
<h2>LLaDA Pretraining</h2>
<h3>Text Pretraining Framework</h3>
</div>

<p align="center">
  <a href="https://github.com/Gen-Verse/MMaDA">
    <img 
        src="https://img.shields.io/badge/Based%20on-MMaDA-green?logo=github&logoColor=white" 
        alt="Based on MMaDA"
    />
  </a>
  <a href="LICENSE">
    <img 
        src="https://img.shields.io/badge/License-MIT-yellow.svg" 
        alt="MIT License"
    />
  </a>
</p>

## 🌟 Introduction
Under testing...（目前还属于团队自用，上传上来的改动了一部分，可能有少许bug，正在测试中）
This is a text pretraining framework for LLaDA models, modified from the [MMaDA](https://github.com/Gen-Verse/MMaDA) codebase.

**Features:**
- Text-only training pipeline
- Distributed training support with DeepSpeed and Accelerate
- YAML-based configuration
- Memory efficient training options

## 🚀 Quick Start

### Env（Source from MMaDA）
```bash
pip install -r requirements.txt
```

### Basic Training
```bash
# Update paths in configs/llada_pretraining.yaml
bash scripts/train.sh
```

## ⚙️ Configuration

Edit `configs/llada_pretraining.yaml`:

```yaml
model:
    pretrained_model_path: ".../LLaDA-8B-Base/"
    # LLaDA specific configuration
    llada_config:
        gradient_checkpointing: false  # close gradient checkpointing
        new_vocab_size: 126464
        # Add other LLaDA specific configs here if needed

dataset:
  params:
    train_shards_path_or_url: "path/to/data"
    
training:
  batch_size: 16
  max_train_steps: 100000
  mixed_precision: "bf16"
```

## 🔧 Training

### Setup Accelerate
```bash
accelerate config
```

You can also use the provided configuration files in `accelerate_configs/` for different hardware and distributed setups:
- `1_gpu.yaml` - Single GPU
- `1_node_only.yaml` - Single node, single process (CPU or GPU)
- `1_node_8_gpus_deepspeed_zero1.yaml` - 8 GPUs with DeepSpeed ZeRO-1
- `1_node_8_gpus_deepspeed_zero2.yaml` - 8 GPUs with DeepSpeed ZeRO-2
- `1_node_8_gpus_deepspeed_zero3.yaml` - 8 GPUs with DeepSpeed ZeRO-3
- `8_node_8_gpus_deepspeed_zero2.yaml` - 8 nodes, each with 8 GPUs, DeepSpeed ZeRO-2

### Run Training
```bash
accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero1.yaml \
    --main_process_port=8888 \
    training/train_llada.py \
    config=configs/llada_pretraining.yaml
```

## 📁 Project Structure

```
LLaDA_pretraining/
├── accelerate_configs/     # Accelerate configurations
├── configs/               # Training configurations
├── models/               # Model implementations
├── parquet/              # Data loading utilities
├── training/             # Training scripts
└── scripts/              # Shell scripts
```

## 🛠️ Data Format
The files under the folder path you provided should be in JSONL format. It is recommended that the dataset be evenly split into multiple files with the number of files greater than the number of GPUs.
```json
{"text": "Training text content"}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Based on [MMaDA](https://github.com/Gen-Verse/MMaDA) by Yang et al.
