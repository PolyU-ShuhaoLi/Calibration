建议CUDA version: CUDA 12.4，能够安装torch2.4.0(RL训练)和torch2.6.0(SFT训练)


1. activate anaconda base环境 

```bash
conda activate
```

2. 

```bash
cd SFT_training/
chmod +x train_sft.sh
source train_sft.sh
```

脚本做了什么：

- creates a conda environment named `finetune`
- installs PyTorch and LlamaFactory
- downloads the dataset from ModelScope
- downloads the model from ModelScope
- launches SFT training with LlamaFactory

SFT输出模型到:

```text
LlamaFactory/saves/Qwen2.5-7B-Instruct/full/train_deepscaler_simplelr_think
```

3. 

```bash
cd RL_training/
chmod +x train_rl_v1.sh
source train_rl_v1.sh
```

脚本做了什么：

- creates/uses a conda environment named `verl`
- installs PyTorch and project dependencies
- clones `simpleRL-reason`
- downloads RL dataset files from ModelScope
- downloads the base model from ModelScope
- starts a local Ray head node
- launches GRPO / RL training


RL模型输出到：
```text
./models
./data
./checkpoints
./logs
```
