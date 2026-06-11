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

# 第二版工程化
可以。从“调用你这个服务的人”的角度，他其实不需要理解 KNN、embedding、PCA 这些细节。他只需要知道三件事：

```text
1. 正常训练数据放在哪里
2. 健康库建在哪里
3. 推理时传入哪个 JSON 文件和哪个设备类型
```

下面是推荐给使用者的操作流程。

**一、准备训练数据**

假设他电脑上有一个设备类型叫：

```text
bearing
```

他需要准备一批“正常状态”的 JSON 音频文件，放到某个目录，例如：

```text
D:\UserData\DCASE\data_train\Default_libraries\bearing\
|-- normal_001.json
|-- normal_002.json
|-- normal_003.json
`-- ...
```

这里注意：

```text
训练目录里应该放正常样本
不要混入明显异常样本
JSON 原始采样率默认 44800Hz
```

如果有多个设备类型，可以这样放：

```text
D:\UserData\DCASE\data_train\Default_libraries\bearing\
D:\UserData\DCASE\data_train\Default_libraries\fan\
D:\UserData\DCASE\data_train\Default_libraries\motor\
```

**二、配置运行路径**

在他的电脑上配置两个路径：

```powershell
$env:DCASE_DATA_TRAIN_ROOT="D:\UserData\DCASE\data_train"
$env:DCASE_LIBRARIES_ROOT="D:\UserData\DCASE\health_libraries"
```

含义是：

```text
DCASE_DATA_TRAIN_ROOT
存放训练 JSON 的根目录

DCASE_LIBRARIES_ROOT
存放构建完成后的健康库
```

默认库名是：

```text
Default_libraries
```

所以程序会自动按这个规则找：

```text
训练数据:
DCASE_DATA_TRAIN_ROOT/Default_libraries/{machine_type}

健康库:
DCASE_LIBRARIES_ROOT/Default_libraries/{machine_type}
```

比如 `machine_type=bearing`：

```text
训练数据目录:
D:\UserData\DCASE\data_train\Default_libraries\bearing

健康库目录:
D:\UserData\DCASE\health_libraries\Default_libraries\bearing
```

**三、建库**

方式 1：显式路径建库，最直观：

```powershell
python build_health_library.py `
  --reference_dir "D:\UserData\DCASE\data_train\Default_libraries\bearing" `
  --library_dir "D:\UserData\DCASE\health_libraries\Default_libraries\bearing" `
  --input_sample_rate 44800 `
  --target_sample_rate 16000 `
  --knn_neighbors 5 `
  --pca_dim 64
```

建库完成后会生成：

```text
D:\UserData\DCASE\health_libraries\Default_libraries\bearing\
|-- metadata.json
|-- manifest.csv
|-- detector.joblib
|-- reference_scores.npy
`-- reference_embeddings.npy
```

方式 2：使用 `machine_type` 自动路径：

```powershell
python build_health_library.py --machine_type bearing
```

前提是已经设置：

```powershell
$env:DCASE_DATA_TRAIN_ROOT="D:\UserData\DCASE\data_train"
$env:DCASE_LIBRARIES_ROOT="D:\UserData\DCASE\health_libraries"
```

并且训练数据在：

```text
D:\UserData\DCASE\data_train\Default_libraries\bearing
```

**四、测试单个文件**

假设新采集到的 JSON 文件是：

```text
D:\UserData\DCASE\incoming\test_001.json
```

命令行方式：

```powershell
python score_file.py `
  --input_path "D:\UserData\DCASE\incoming\test_001.json" `
  --machine_type bearing
```

程序会自动加载：

```text
D:\UserData\DCASE\health_libraries\Default_libraries\bearing
```

输出类似：

```json
{
  "status": "ok",
  "detector_type": "knn",
  "raw_knn_distance": 20.85,
  "abnormality_score": 44.0,
  "risk_level": "normal_like",
  "health_score": 100.0,
  "health_level": "healthy"
}
```

业务上主要看：

```text
health_score
health_level
```

例如：

```text
healthy             健康
slightly_degraded   轻微退化
warning             预警
unhealthy           不健康
critical            严重异常
```

**五、调用 Python 服务接口**

如果他的程序是 Python，可以这样调用：

```python
from dcase_ae.service import analyze_file

result = analyze_file(
    input_path=r"D:\UserData\DCASE\incoming\test_001.json",
    machine_type="bearing",
)

print(result)
```

返回的是一个 `dict`，里面包含：

```python
result["status"]
result["health_score"]
result["health_level"]
result["abnormality_score"]
result["raw_knn_distance"]
```

这是后续集成到他的采集程序里最推荐的方式。

**六、批量测试一批文件**

当前服务接口是单文件分析。如果他有一批 JSON 文件，可以在自己的程序里循环调用：

```python
from pathlib import Path
from dcase_ae.service import analyze_file

machine_type = "bearing"
input_dir = Path(r"D:\UserData\DCASE\incoming_batch")

for path in input_dir.rglob("*.json"):
    result = analyze_file(path, machine_type)
    print(path.name, result["health_score"], result["health_level"])
```

**七、如果健康库不存在怎么办**

现在有两种模式。

默认模式：健康库不存在就报错。

也就是说，如果还没建库，调用：

```python
analyze_file("xxx.json", "bearing")
```

会返回类似：

```json
{
  "status": "error",
  "status_code": "health_library_incomplete",
  "message": "..."
}
```

如果你想允许“第一次推理时自动建库”，需要配置：

```powershell
$env:DCASE_ENABLE_LIBRARY_TRAINING="1"
```

然后确保训练数据目录存在：

```text
D:\UserData\DCASE\data_train\Default_libraries\bearing
```

不过我建议生产使用时还是显式先建库，不要第一次推理自动训练。这样更可控。

**八、给使用者的一套最简说明**

你可以这样告诉他：

```text
1. 把正常训练 JSON 放到：
   D:\UserData\DCASE\data_train\Default_libraries\bearing

2. 设置环境变量：
   DCASE_DATA_TRAIN_ROOT=D:\UserData\DCASE\data_train
   DCASE_LIBRARIES_ROOT=D:\UserData\DCASE\health_libraries

3. 建库：
   python build_health_library.py --machine_type bearing

4. 推理：
   python score_file.py --input_path D:\UserData\DCASE\incoming\test_001.json --machine_type bearing

5. Python 调用：
   from dcase_ae.service import analyze_file
   result = analyze_file("D:/UserData/DCASE/incoming/test_001.json", "bearing")
```

一句话总结：**用户提供正常训练 JSON 目录，你用 `build_health_library.py` 为每种设备类型建一个健康库；建库完成后，他只需要用 `machine_type + input_path` 调用你的推理接口。**