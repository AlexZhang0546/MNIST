# PyTorch MNIST 手写数字识别

这是一个面向初学者的完整 MNIST 项目。训练脚本会在第一次运行时自动下载 60000 张训练图片和 10000 张测试图片，并训练一个约 46.8 万参数的卷积神经网络（CNN）。默认训练 5 轮，正常环境中的测试准确率通常可以达到 99% 左右，满足不低于 98% 的要求。

> 注意：准确率取决于依赖版本和数据是否完整。本项目会在训练结束后读取最佳模型并在独立测试集上复核；只有复核准确率达到 98%，程序才返回成功状态。

## 1. 创建环境

建议使用 Python 3.10 或更高版本。下面的命令会创建独立虚拟环境，避免污染系统 Python：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果需要指定 CUDA 或 CPU 版本的 PyTorch，请按照 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 提供的命令安装。本项目也会在没有 GPU 时自动使用 CPU。

## 2. 训练

```bash
python train.py
```

首次运行时，`torchvision.datasets.MNIST(..., download=True)` 会自动下载数据到 `data/`。训练完成后会生成：

- `outputs/best_model.pth`：验证准确率最高的模型参数。
- `outputs/history.json`：每一轮的训练损失、训练准确率、验证损失、验证准确率和耗时。
- `outputs/metrics.json`：最终复核准确率、最佳轮次、总训练时间和是否达到目标。

可以把训练轮数改多，例如：

```bash
python train.py --epochs 8 --batch-size 64 --learning-rate 0.001
```

常用参数：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--epochs` | 5 | 整个训练集被学习多少轮 |
| `--batch-size` | 64 | 每次送入网络多少张图片 |
| `--learning-rate` | 0.001 | Adam 优化器更新参数的步长 |
| `--workers` | 2 | 后台读取数据的进程数 |
| `--seed` | 42 | 随机种子，便于复现实验 |
| `--target-accuracy` | 0.98 | 训练结束后的达标阈值 |

## 3. 图形化训练与推理

项目提供 Tkinter 桌面界面，不需要额外安装 GUI 依赖：

```bash
source .venv/bin/activate
python app.py
```

界面左侧可以调整训练轮数、批次大小、学习率、数据读取进程数、随机种子和目标准确率。点击“开始训练”后，训练脚本会在后台进程中运行，界面不会卡死；进度条、训练日志和最终测试准确率会实时更新。

界面右侧是一个 280×280 的正方形画布。用鼠标在黑底画布上写出数字，可以调整画笔粗细、清空画布并点击“识别数字”。模型会显示预测数字、置信度以及概率最高的三个候选数字。每次训练完成后，再次识别时会自动加载最新的模型文件。

如果 Linux 系统提示缺少 Tkinter，可安装对应的系统包，例如 Ubuntu：

```bash
sudo apt install python3-tk
```

## 4. 使用图片预测

MNIST 的风格是黑底白字。准备一张手写数字图片，然后运行：

```bash
python predict.py 图片路径.jpg
```

如果图片是白底黑字，常用扫描件或手机拍照图片多属于这种情况，需要加入 `--invert`：

```bash
python predict.py 图片路径.jpg --invert
```

预测脚本会显示最可能的数字，以及概率最高的三个候选结果。训练集图片本身可以从 `data/MNIST/raw/train-images-idx3-ubyte` 解析，但最简单的方法是直接使用 `predict.py` 处理独立的 PNG 或 JPG 图片。

也可以直接从 MNIST 测试集选择一个带真实标签的样本，用于验证推理流程：

```bash
python predict.py --sample-index 0
```

一次识别多张图片并保存 JSON 结果：

```bash
python predict.py 图片1.png 图片2.png --top-k 3 --output-json outputs/predictions.json
```

`predict.py` 会自动修正手机照片的 EXIF 旋转信息、转换为灰度图、等比例缩放并居中到 28×28 黑色画布。普通白底黑字图片需要加 `--invert`，MNIST 黑底白字图片不需要反转。

## 5. CNN 是如何识别数字的

MNIST 图片是 28×28 的灰度图。项目先把每张图片变成形状为 `(1, 28, 28)` 的张量，再用训练集的均值 `0.1307` 和标准差 `0.3081` 做标准化。

网络分为两部分：

1. 特征提取：四次卷积和两次最大池化逐渐把图像从 `28×28` 缩小到 `7×7`，同时把通道数从 1 增加到 64。卷积核负责学习笔画、圆弧和交叉点等特征。
2. 分类：展平 64 个 `7×7` 特征图，经过全连接层得到 10 个分数，分别对应数字 0 到 9。

训练一次时，程序按以下顺序工作：

1. 前向传播：网络给每张图片输出 10 个类别分数。
2. 计算损失：交叉熵损失衡量预测结果与真实标签的差距。
3. 反向传播：自动求导计算每个参数的梯度。
4. 参数更新：Adam 优化器根据梯度调整卷积核和全连接层权重。

Dropout 会在训练时随机屏蔽部分神经元或特征图，使网络不能只依赖少量特征做判断；评估和预测时会自动关闭 Dropout。

## 6. FPGA 移植教程

如果希望把训练好的 `DigitCNN` 作为神经网络 FPGA 部署的验证模型，并进一步研究“混沌加密信号 → 神经网络密钥/解密运算 → 视频恢复”的硬件流水线，请阅读 [FPGA 三种成熟路线教程](docs/FPGA_DEPLOYMENT_GUIDE.md)。教程保留 hls4ml、FINN + Brevitas 和 Vitis AI + DPU 三条路线，按模型能力、资源、接口和运行时选型，不预设具体 FPGA 板卡，并覆盖信号输入契约、BER/PSNR、持续视频吞吐和序列同步。

## 7. 为什么用测试集单独评估

训练集用于更新模型参数，测试集只用于最终检查泛化能力。如果只看训练准确率，模型可能只是背下训练样本，也就是过拟合。本项目先从原始训练集随机划出 90% 用于训练、10% 用作验证集，每轮根据验证准确率保存表现最好的一轮参数。10000 张官方测试图片不参与训练或选模，只在全部训练结束后使用一次。

## 8. 常见问题

**没有网络，程序无法下载数据。** 请让第一次运行训练脚本的环境能访问 `ossci-datasets.s3.amazonaws.com`，或提前把 MNIST 的四个原始文件放到 `data/MNIST/raw/`。

**运行提示 `ModuleNotFoundError: No module named 'torch'`。** 说明当前 Python 没有安装 PyTorch，或者没有激活 `.venv`。先执行 `source .venv/bin/activate`，再重新安装 `requirements.txt`。

**训练完成但准确率不到 98%。** 先确认使用的是默认模型和完整数据，再尝试 `python train.py --epochs 8`。不要把测试集也用于参数更新，否则报告结果会失去意义。

**CPU 训练太慢。** CNN 很小，默认参数通常仍能在几分钟内完成；`--workers 0` 可排除某些系统上的多进程数据读取问题，有 CUDA 时会自动使用 GPU。
