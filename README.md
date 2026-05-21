# DCASE 2023 Task 2 Baseline AE, Simplified

This is a minimal PyTorch refactor of the DCASE Task 2 AutoEncoder baseline.
It keeps the original `AENet` implementation in `networks/dcase2023t2_ae/network.py`,
but removes cross-year dataset handling, shell scripts, auto-download code, YAML configuration,
and challenge submission formatting.

## Project Layout

```text
.
|-- train.py
|-- test.py
|-- configs.py
|-- dcase_ae/
|   |-- audio.py
|   |-- features.py
|   |-- dataset.py
|   |-- trainer.py
|   |-- evaluator.py
|   |-- metrics.py
|   `-- utils.py
|-- networks/
|   `-- dcase2023t2_ae/
|       `-- network.py
|-- checkpoints/
`-- outputs/
```

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

The test script always writes file-level anomaly scores:

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
