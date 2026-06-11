当前项目可以理解成一个 **基于预训练音频 embedding + KNN 的设备异常/健康度检测 baseline**。

**整体目标**

你的输入原始是 JSON 音频数据，采样率约 44.8kHz。项目最终要做的是：

```text
JSON 音频
-> 转成 16kHz 音频
-> 用预训练音频模型提 embedding
-> 用 KNN 和正常样本库比较
-> 输出异常程度 / 风险等级
```

目前核心不是训练一个深度模型，而是：

```text
用预训练模型做特征提取
用正常样本 embedding 建立 reference bank
新样本和 reference bank 比距离
```

**数据流程**

原始 JSON 数据建议放：

```text
data/data_json/bearing/
|-- train/
|   |-- normal_0001.json
|   |-- normal_0002.json
|   `-- ...
`-- test/
    |-- normal_xxx.json
    |-- anomaly_xxx.json
    `-- ...
```

其中 `train/` 主要放正常数据。

然后用：

```powershell
python prepare_json_audio.py `
  --input_dir .\data\data_json\bearing `
  --output_dir .\data\data_wav16k\bearing `
  --input_sample_rate 44800 `
  --target_sample_rate 16000
```

转换成：

```text
data/data_wav16k/bearing/
|-- train/
|   |-- normal_0001.wav
|   `-- ...
`-- test/
    |-- normal_xxx.wav
    |-- anomaly_xxx.wav
    `-- ...
```

**训练流程**

训练入口是：

```powershell
python train.py `
  --data_dir .\data\data_wav16k\bearing `
  --checkpoint_path .\checkpoints\knn.joblib
```

训练做的事情是：

```text
读取 train/*.wav
-> 预训练模型提 embedding
-> StandardScaler 标准化
-> PCA 降到 64 维
-> 保存正常样本 embedding bank
-> 计算 leave-one-out KNN reference scores
-> 保存 checkpoints/knn.joblib
```

`knn.joblib` 里面保存的核心内容包括：

```text
KNN detector
normal embeddings
scaler
pca
normal_reference_scores
训练配置
```

**推理流程**

单个 JSON 推理入口是：

```powershell
python score_file.py `
  --input_path .\data\data_json\bearing\test\xxx.json `
  --checkpoint_path .\checkpoints\knn.joblib
```

它会：

```text
读取 JSON
-> 默认按 44.8kHz
-> 下采样到 16kHz
-> 提 embedding
-> 用训练时的 scaler/PCA 转换
-> 和正常 embedding bank 算 KNN 距离
-> 输出 JSON 结果
```

输出里目前关键字段是：

```json
{
  "raw_knn_distance": 20.85,
  "abnormality_score": 44.0,
  "risk_level": "normal_like"
}
```

解释：

```text
raw_knn_distance 越大，越远离正常样本
abnormality_score 是异常百分位，0-100，越大越异常
risk_level 是按 90/95/99 阈值给出的风险等级
```

**批量测试流程**

批量测试入口是：

```powershell
python test.py `
  --data_dir .\data\data_wav16k\bearing `
  --checkpoint_path .\checkpoints\knn.joblib
```

它读取：

```text
data/data_wav16k/bearing/test/*.wav
```

输出：

```text
outputs/scores.csv
outputs/metrics.csv
```

适合评估正常/异常整体区分效果。

**当前方法的核心逻辑**

KNN 检测器的判断逻辑是：

```text
新样本 embedding
-> 找最近的 k 个正常 embedding
-> 计算平均距离
-> 距离越大，越不像正常
```

然后用训练正常样本自己的 reference scores 做比较：

```text
新样本距离超过多少比例的正常 reference scores
= abnormality_score
```

比如：

```text
abnormality_score = 96
```

表示：

```text
这个样本比 96% 的训练正常样本更偏离正常
```

**你现在还想加的健康度**

当前已有的是异常百分位，不是健康度。后续比较适合新增：

```json
{
  "health_score": 96.5,
  "health_level": "healthy"
}
```

推荐逻辑是：

```text
低于正常 p50: 健康度 100
p50-p90: 100 降到 95
p90-p95: 95 降到 90
p95-p99: 90 快速降到 40
超过 p99: 继续降到 0
```

这样正常样本即使 `abnormality_score=44`，健康度仍然可以保持在 90-100，更符合设备健康评分直觉。

一句话总结：  
这个项目现在是 **JSON 音频 -> 16kHz -> 预训练 embedding -> KNN 正常参考库 -> 异常百分位输出** 的异常检测系统。下一步最自然的是在单文件推理输出里增加 `health_score`。