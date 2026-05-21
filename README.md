# DCASE Task 2 Baseline, Pretrained Embedding + GMM

This refactor keeps the existing local data flow and evaluation outputs, but replaces the
AutoEncoder training/scoring core with:

```text
pretrained audio model -> mean-pooled embedding -> GaussianMixture score
```

The pretrained model is frozen and used only as an embedding extractor. The default model is
`microsoft/wavlm-base`.

## Dataset

Prepare one local dataset directory:

```text
your_dataset/
|-- train/
|   |-- normal_0000.wav
|   `-- ...
`-- test/
    |-- section_00_source_test_normal_0000.wav
    |-- section_00_source_test_anomaly_0000.wav
    |-- section_00_target_test_normal_0000.wav
    `-- section_00_target_test_anomaly_0000.wav
```

The code does not download DCASE data and does not loop over machine types. Pass the exact
dataset directory you want to train and score.

## Train

```bash
python train.py --data_dir /path/to/your_dataset
```

Training now means:

```text
read train wav files
extract one embedding per wav with the frozen pretrained model
fit sklearn.mixture.GaussianMixture on the train embeddings
save the GMM with joblib
```

The default checkpoint is:

```text
checkpoints/gmm.joblib
```

Useful options:

```bash
python train.py \
  --data_dir /path/to/your_dataset \
  --pretrained_model_name microsoft/wavlm-base \
  --embedding_sample_rate 16000 \
  --batch_size 8 \
  --gmm_components 4
```

## Test

```bash
python test.py --data_dir /path/to/your_dataset --checkpoint_path checkpoints/gmm.joblib
```

Testing now means:

```text
read test wav files
extract one embedding per wav
score each embedding with gmm.score_samples
use negative log likelihood as anomaly_score
write scores.csv and metrics.csv
```

The test script always writes:

```text
outputs/scores.csv
```

Each row contains:

```text
filename,anomaly_score,label,domain
```

If filenames contain `normal` / `anomaly`, metrics are also written:

```text
outputs/metrics.csv
```

Metric labels are parsed from the filename:

```text
normal   -> label 0
anomaly  -> label 1
source   -> source domain
target   -> target domain
```

`metrics.csv` columns:

```text
AUC (source),AUC (target),pAUC,pAUC (source),pAUC (target),
precision (source),precision (target),recall (source),recall (target),
F1 score (source),F1 score (target)
```

If a required group has only normal or only anomaly samples, that metric is written as `nan`.

## Notes

The first run of `microsoft/wavlm-base` requires the model to be available from Hugging Face or
already cached locally. To use another compatible Hugging Face audio model, pass
`--pretrained_model_name`.



## 增加BEATs与GMM打分

已完成：核心算法已经从 **AutoEncoder + MSE** 替换成了 **预训练音频大模型 embedding + GMM 负对数似然打分**。

改动重点：

- 新增 [dcase_ae/embedder.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/dcase_ae/embedder.py)
  - 使用 `transformers` 加载默认模型 `microsoft/wavlm-base`
  - 冻结所有参数
  - 对 `last_hidden_state` 做 mean pooling，得到每条 wav 一个 embedding

- 重写 [dcase_ae/trainer.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/dcase_ae/trainer.py)
  - 不再训练 AE
  - 不再有 optimizer / epoch / loss / backward
  - 遍历 `train/`，提取 embedding
  - 使用 `sklearn.mixture.GaussianMixture` 拟合
  - 保存到 `checkpoints/gmm.joblib`

- 修改 [dcase_ae/evaluator.py](D:/CodeMater/DCASE/myproject/dcase2023_task2_baseline_ae/dcase_ae/evaluator.py)
  - 加载 GMM
  - 遍历 `test/`，提取 embedding
  - 使用：

```python
anomaly_score = -gmm.score_samples(embedding)
```

  - 后续 `scores.csv`、`metrics.csv`、AUC/pAUC/precision/recall/F1 的保存逻辑保留原来的输出形式

新增依赖：

```text
transformers
scikit-learn
joblib
```

现在使用方式：

```bash
python train.py --data_dir /path/to/your_dataset
python test.py --data_dir /path/to/your_dataset --checkpoint_path checkpoints/gmm.joblib
```

可调参数示例：

```bash
python train.py \
  --data_dir /path/to/your_dataset \
  --pretrained_model_name microsoft/wavlm-base \
  --embedding_sample_rate 16000 \
  --batch_size 8 \
  --gmm_components 4
```

我也做了静态检查：

```bash
python train.py --help
python test.py --help
python -m compileall train.py test.py configs.py dcase_ae
```

都通过了。实际加载 `microsoft/wavlm-base` 需要你的服务器能访问 HuggingFace，或者模型已经在本地缓存里。