# DCASE 2023 Task 2 Baseline AE, Simplified

This is a minimal PyTorch refactor of the DCASE Task 2 AutoEncoder baseline.
It keeps the original `AENet` implementation in `networks/dcase2023t2_ae/network.py`,
but removes cross-year dataset handling, shell scripts, auto-download code, YAML configuration,
and challenge submission formatting.

## Project Layout

```text
.
├── train.py
├── test.py
├── configs.py
├── dcase_ae/
│   ├── audio.py
│   ├── features.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── utils.py
├── networks/
│   └── dcase2023t2_ae/
│       └── network.py
├── checkpoints/
└── outputs/
```

## Dataset

Prepare one local dataset directory with this structure:

```text
your_dataset/
├── train/
│   ├── normal_0000.wav
│   └── ...
└── test/
    ├── sample_0000.wav
    └── ...
```

The code does not download anything and does not loop over machine types. Pass the exact dataset
directory you want to train and score.

## Train

```bash
python train.py --data_dir /path/to/your_dataset
```

The default checkpoint is written to:

```text
checkpoints/ae_model.pt
```

## Test

```bash
python test.py --data_dir /path/to/your_dataset --checkpoint_path checkpoints/ae_model.pt
```

The default score file is:

```text
outputs/scores.csv
```

Each row contains:

```text
filename,anomaly_score
```

## Defaults

The feature defaults follow the original baseline:

```text
n_mels=128
frames=5
n_fft=1024
hop_length=512
power=2.0
```

The network architecture is not redesigned. `AENet` is still the original core model.



## 重构后使用指南
 已完成第一版极简重构，核心 AE 网络文件保持未改动：[network.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/networks/dcase2023t2_ae/network.py)。

现在项目主结构已经收敛为：

```text
train.py
test.py
configs.py
dcase_ae/
networks/dcase2023t2_ae/network.py
checkpoints/
outputs/
README.md
requirements.txt
```

我完成了这些事：

- 新增本地数据读取流程：只要求 `data_dir/train/*.wav` 和 `data_dir/test/*.wav`
- 新增 `dcase_ae/audio.py`、`features.py`、`dataset.py`、`trainer.py`、`evaluator.py`、`utils.py`
- 重写 [train.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/train.py) 和 [test.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/test.py)
- 删除旧 `.sh`、`datasets/`、`tools/`、复杂 YAML、跨年份包装器、提交格式化相关代码
- 更新 [README.md](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/README.md)
- 新增 `.gitignore`，本地 `data/`、旧 `models/`、`results/`、`logs/` 会被忽略，不会被新项目依赖

验证做了轻量检查：

```bash
python -m compileall train.py test.py configs.py dcase_ae networks
python train.py --help
python test.py --help
```

都通过了。没有跑真实训练，因为当前环境缺少 `torch`，也符合你说的本地环境限制。

现在使用方式是：

```bash
python train.py --data_dir /path/to/your_dataset
python test.py --data_dir /path/to/your_dataset --checkpoint_path checkpoints/ae_model.pt
```

#### 重构后的训练步骤
当然。这个重构后的项目可以理解为：**保留原始 AutoEncoder 网络，重写数据、训练、测试这三条主线，把 DCASE 官方 baseline 变成一个本地单数据集可用的最小 PyTorch 项目。**

**整体目标**
原项目做了很多事情：跨年份、跨 machine type、自动下载、比赛提交格式、官方评估结果汇总、复杂 yaml 配置等。现在这些都被移除了。

现在项目只做一件事：

```text
给定一个本地数据目录
读取 train/ 和 test/ 下的 wav
提取 log-mel 特征
训练 AE
对 test wav 输出 anomaly score
```

核心目录现在是：

```text
.
├── train.py
├── test.py
├── configs.py
├── dcase_ae/
│   ├── audio.py
│   ├── features.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── utils.py
├── networks/
│   └── dcase2023t2_ae/
│       └── network.py
├── checkpoints/
└── outputs/
```

**数据格式**
现在只支持一个本地数据集路径，要求结构是：

```text
your_dataset/
├── train/
│   ├── xxx.wav
│   └── ...
└── test/
    ├── yyy.wav
    └── ...
```

不再要求：

```text
DCASE2023T2bearing
section_00
source / target
dev_data / eval_data
attributes_00.csv
eval_data_list_xxxx.csv
```

这些复杂比赛结构现在都不参与代码逻辑。

**入口文件**
[train.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/train.py) 是训练入口。

典型命令：

```bash
python train.py --data_dir /path/to/your_dataset
```

它做的事很薄：

1. 解析命令行参数
2. 构造 `Config`
3. 设置随机种子
4. 调用 `AETrainer(cfg).fit()`
5. 保存 checkpoint 到 `checkpoints/ae_model.pt`

[test.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/test.py) 是测试入口。

典型命令：

```bash
python test.py --data_dir /path/to/your_dataset --checkpoint_path checkpoints/ae_model.pt
```

它会读取 `test/` 下每个 wav，输出：

```text
outputs/scores.csv
```

格式是：

```csv
filename,anomaly_score
xxx.wav,0.1234567890
```

不再输出比赛提交用的 `anomaly_score_xxx_section_xx_test_seedxxx.csv`、`decision_result_xxx.csv`、`result_xxx_roc.csv` 等文件。

**配置文件**
[configs.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/configs.py) 用一个 `Config` dataclass 替代原来的 `baseline.yaml`。

里面包含三类配置：

音频特征参数：

```python
n_mels = 128
frames = 5
n_fft = 1024
hop_length = 512
power = 2.0
```

训练参数：

```python
batch_size = 256
epochs = 100
learning_rate = 1e-3
weight_decay = 1e-4
```

模型参数：

```python
hidden_dim = 512
latent_dim = 64
dropout = 0.1
```

这些默认值基本沿用了原 baseline 里的参数，但配置方式更直接，不再走 yaml -> argparse 的绕路。

**音频读取**
[dcase_ae/audio.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/dcase_ae/audio.py) 只负责读取 wav。

它使用 `librosa.load()`：

```python
y, sr = librosa.load(path, sr=None, mono=mono)
```

注意这里 `sr=None`，意思是保留原始采样率，不强制重采样。

如果 `mono=False` 且读到多通道，它取第一个通道。这基本对应原 baseline 对多通道音频只取一个通道的习惯。

**特征提取**
[dcase_ae/features.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/dcase_ae/features.py) 是非常关键的文件。

它做的是原 baseline 的经典流程：

1. wav -> mel spectrogram
2. mel -> log-mel
3. 连续多帧拼接成一个向量

也就是：

```text
wav
-> mel spectrogram: shape = [n_mels, time]
-> log mel
-> frame stacking
-> vectors: shape = [num_vectors, n_mels * frames]
```

默认情况下：

```text
n_mels = 128
frames = 5
input_dim = 128 * 5 = 640
```

所以每个训练样本是一个 640 维向量。

这里保留了原 baseline 的 log 公式：

```python
20.0 / power * np.log10(...)
```

这个细节很重要，因为它影响特征数值尺度。

**数据集封装**
[dcase_ae/dataset.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/dcase_ae/dataset.py) 是新数据流的核心。

里面有三个主要类：

`LocalDCASEFrameDataset`

用于训练。它会读取 `train/` 下所有 wav，把每个 wav 提取成很多帧级向量，然后全部拼成一个大的训练集：

```text
train wav files
-> many feature vectors
-> Dataset[index] returns one vector
```

所以训练时 batch 是：

```text
[batch_size, input_dim]
```

例如：

```text
[256, 640]
```

`LocalDCASEFileDataset`

用于测试。它不会把所有文件拆散，而是保留“一个 wav 文件对应一组特征向量”的关系：

```text
test wav
-> vectors: [num_vectors, input_dim]
```

这样 evaluator 可以对一个 wav 的所有 frame score 聚合成一个 file-level anomaly score。

`LocalDCASEDataModule`

它负责把 `train/` 和 `test/` 组织起来，并提供：

```python
train_valid_loaders()
test_loader()
```

训练集会按 `validation_split` 切分一部分作为验证集。

**训练逻辑**
[dcase_ae/trainer.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/dcase_ae/trainer.py) 负责训练 AE。

它使用的模型是：

```python
AENet(
    input_dim=cfg.input_dim,
    block_size=cfg.n_mels,
    num_conditions=0,
    latent_dim=cfg.latent_dim,
    hidden_dim=cfg.hidden_dim,
    dropout=cfg.dropout,
)
```

注意这里：

```python
num_conditions=0
```

这是因为新项目不再区分 section id / machine id / source-target domain，所以不再使用条件向量。

训练 loss 使用：

```python
F.smooth_l1_loss(recon, data)
```

这是沿用你当前原项目里较新的训练方式。验证时用 MSE 计算 reconstruction error。

训练期间会保存验证集 loss 最低的模型到：

```text
checkpoints/ae_model.pt
```

checkpoint 里包含：

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "config": config,
}
```

**测试逻辑**
[dcase_ae/evaluator.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/dcase_ae/evaluator.py) 负责打分。

流程是：

```text
读取 checkpoint
加载 AENet
遍历 test/ 下每个 wav
提取该 wav 的所有 feature vectors
AE 重建
计算每个 frame 的 MSE
聚合为一个 wav 的 anomaly score
写入 outputs/scores.csv
```

frame-level score 是：

```python
F.mse_loss(recon, vectors, reduction="none").mean(dim=1)
```

file-level score 使用一个“均值 + 高分位尾部”的聚合方式：

```text
score = (1 - tail_weight) * mean_score + tail_weight * quantile_score
```

默认是：

```text
file_score_quantile = 0.9
file_score_tail_weight = 0.5
```

也就是一半看整体平均异常程度，一半看最高 10% 那部分的异常程度。

**网络结构**
[networks/dcase2023t2_ae/network.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/networks/dcase2023t2_ae/network.py) 是保留的核心网络。

这个文件没有改结构。

里面主要是：

```python
ResidualBlock
AENet
```

`AENet` 的大致结构是：

```text
input
-> encoder
-> latent z
-> decoder
-> reconstruction
```

encoder：

```text
Linear(input_dim -> hidden_dim)
LayerNorm
SiLU
Dropout
ResidualBlock
Linear(hidden_dim -> hidden_dim // 2)
LayerNorm
SiLU
Dropout
ResidualBlock
Linear(hidden_dim // 2 -> latent_dim)
LayerNorm
```

decoder 反过来：

```text
Linear(latent_dim -> hidden_dim // 2)
...
Linear(hidden_dim -> input_dim)
```

这部分是整个项目最重要的“不动点”。

**utils**
[dcase_ae/utils.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/dcase_ae/utils.py) 放了一些小工具：

```python
set_seed()
get_device()
ensure_dir()
save_checkpoint()
load_checkpoint()
```

主要是为了让 trainer/evaluator 更干净。

**当前项目删掉了什么**
已经删除或不再使用：

```text
*.sh
baseline.yaml
common.py
datasets/
tools/
networks/base_model.py
networks/models.py
networks/criterion/
networks/dcase2023t2_ae/dcase2023t2_ae.py
README_legacy.md
```

这些旧文件主要负责：

```text
跨年份兼容
自动下载
machine type 遍历
section id 处理
source/target 条件向量
比赛提交格式
AUC/pAUC/decision result 汇总
画图
旧结果导出
```

都不符合你这次“单数据路径、本地化、极简”的目标，所以移除了。

**当前版本的边界**
这个版本非常干净，但也有明确取舍：

1. 不再支持多个 machine type 自动循环。
2. 不再自动识别 section/source/target。
3. 不再计算 AUC、pAUC、F1，因为测试数据可能没有标签。
4. 不再输出 DCASE 官方提交格式。
5. 不再缓存 mel 特征到 pickle，每次运行会重新提特征。
6. 只保留最核心 AE reconstruction-based anomaly score。

换句话说，它现在更像一个“研究/实验用的干净 AE baseline”，而不是 DCASE 官方比赛全流程工具箱。

**最小使用路径**
你现在只需要准备：

```text
my_data/
├── train/
│   ├── a.wav
│   ├── b.wav
│   └── ...
└── test/
    ├── x.wav
    ├── y.wav
    └── ...
```

然后：

```bash
python train.py --data_dir my_data
python test.py --data_dir my_data
```

结果在：

```text
outputs/scores.csv
```

这一版的重构方向已经比较稳：**网络不动，数据流变简单，训练/测试分离，配置显式化，比赛工程负担全部拿掉。**