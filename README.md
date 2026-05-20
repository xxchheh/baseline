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