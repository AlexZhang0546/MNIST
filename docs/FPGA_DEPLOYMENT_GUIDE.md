# 将神经网络“密钥模型”移植到 FPGA：面向混沌加密信号转视频的技术初探

> 适用项目：[`/home/alex/Documents/program/MINIST`](/home/alex/Documents/program/MINIST)  
> 文档对象：当前仓库的 `DigitCNN` 是 FPGA 部署可行性验证模型，最终项目则需要部署参与混沌解密、作用类似密钥的神经网络模型  
> 检索与整理日期：2026-10-31  
> 目标：比较三条成熟的 FPGA 部署路线，为“加密混沌信号 → 神经网络密钥/解密运算 → 视频帧恢复”的实时流水线选择技术底座

**快速导航**

- [先看结论与器件中立选型](#0-先看结论)
- [验证模型与最终模型映射](#1-验证模型与部署约束)
- [路线一：hls4ml](#2-路线一hls4ml-生成-hls-神经网络-ip)
- [路线二：FINN + Brevitas](#3-路线二finn--brevitas-数据流加速器)
- [路线三：Vitis AI + DPU](#4-路线三vitis-ai-quantizer--dpu--vart)
- [三方案比较](#5-三种方案的比较与选择方法)
- [控制预览界面与 FPGA 服务集成](#6-控制预览界面与-fpga-的集成)
- [上板验收与排错](#7-建议的验收流程)
- [官方参考资料](#9-参考资料与版本说明)

## 0. 先看结论

FPGA 并没有一种“对所有项目都最好”的神经网络部署方法。当前 `DigitCNN` 输入为 `1×28×28`、约 13.27M 次乘法累加，可作为工具链验证模型；最终项目的关键指标则会转为解密误码率、输出确定性、持续吞吐、序列同步和视频恢复质量。下面三条路线仍是综合官方文档完整度、活跃版本、真实板卡例程数量、量化支持和模型可改造程度后选出的最成熟方案。

| 路线 | 最适合的目标 | 目标器件要求 | 量化方式 | 移植难度 | 相对优势 | 主要限制 |
| --- | --- | --- | --- | --- | --- | --- |
| **hls4ml** | 用纯 CNN 后端快速制作信号处理原型并评估定点误差 | 能运行 Vivado HLS；可提供 AXI/DMA 接口；资源和时序满足实现结果 | 先 `ap_fixed`，再可接 QONNX 量化 | 中 | Python API 最接近现有项目，CNN 层支持明确，便于逐层观察解密误差 | 必须经过 `torch.fx` 追踪；长序列、状态保持和复杂控制流支持有限 |
| **FINN + Brevitas** | 追求低比特、持续低延迟的固定 CNN 数据流 | 有足够 BRAM/DSP/逻辑；支持 AXI Stream；能做 Vivado 综合和板级封装 | 1–8 bit QAT 或 PTQ，通常从 4W4A 起步 | 高 | 适合持续送入加密样本，自动生成流式硬件并折叠并行度 | 模型拓扑必须符合 FINN；对递归状态、动态长度和通用视频处理并不友好 |
| **Vitis AI + DPU** | 神经网络只是图像/视频处理链中的一个 CNN 模块，后续还要接摄像头与显示 | 目标平台有匹配的 DPU 位流、驱动、运行时、镜像和目标 `arch.json` | DPU INT8，源码量化加板端运行 | 中高 | 运行时、目标平台镜像和视频平台流程成熟，适合做成持续服务 | DPU 只执行受支持的 CNN 图；混沌状态迭代、同步和最终解扰通常仍需 PL/CPU 实现 |

本文不预设具体板卡，也不需要先决定厂商型号。若当前只要求“验证神经网络后端能否上板”，先走 **hls4ml**；若加密样本会持续从 AXI Stream 输入且核心指标是稳定样本级延迟，优先评估 **FINN**；若目标平台已经有匹配的 DPU 位流与运行时，并且神经网络只占视频处理链的一部分，优先评估 **Vitis AI**。最终选择应由器件资源、接口、时序、功耗和供应情况决定，而不是反向迁就示例板卡。

不绑定具体板卡的能力评估表：

| 目标平台能力 | 优先评估 | 注意事项 |
| --- | --- | --- |
| 纯 FPGA 或自定义板卡 | hls4ml 普通 HLS IP；FINN 生成的数据流 IP | 需要自行完成 AXI Stream、AXI-Lite、DMA、时钟复位和主机驱动 |
| SoC FPGA（PS + PL） | 三条都可评估 | 先划分视频协议/解扰由 PS 还是 PL 完成；DPU 路线还要求匹配 SoC 平台 |
| PCIe 加速卡 | hls4ml 或 FINN | 重点评估主机传输、批处理深度、排队和端到端帧延迟 |
| 已提供 DPU 平台的器件 | Vitis AI + DPU | 必须取得匹配的 DPU 位流、驱动、运行时和目标架构文件 |

建议的选型顺序是：**确认网络结构与精度 → 确认信号接口和持续吞吐 → 确认所需 IP/运行时 → 评估器件容量、时序与功耗 → 最后选择具体板卡**。三条路线的成熟实现目前主要建立在 AMD/Xilinx 工具生态上；如果考虑其他厂商 FPGA，hls4ml 生成的 HLS C++ 可能部分复用，但 FINN 和 Vitis AI/DPU 不能视为直接可移植，需要重新评估对应厂商的神经网络工具链。

本文后续出现具体板卡、驱动或目录名时，仅表示官方示例的可复现配置，不是推荐结论。真正实施时才把目标 `part`、板卡标识、时钟约束和平台镜像填入配置。

### 0.1 本项目默认推荐的验收方式

训练仍然在 PC 或 GPU 服务器完成，FPGA 负责解密链中的神经网络推理。最终项目有两种可能结构：神经网络直接输出恢复后的信号，或输出密钥流/解密参数，再由可编程逻辑或 CPU 执行解扰。无论采用哪一种，测试时都必须覆盖完整视频链路：

```mermaid
flowchart LR
    A[混沌加密样本流] --> B[帧同步与序列编号]
    B --> C[定点/浮点输入适配]
    C --> D[FPGA 神经网络密钥模型]
    D --> E[恢复信号或密钥参数]
    E --> F[PL/CPU 解扰与逆变换]
    F --> G[帧缓冲与视频编码]
    G --> H[视频输出与预览界面]
```

当前 `DigitCNN` 仍输出 10 个分类分数，这只是为了验证“训练 → 量化 → 综合 → 上板 → 数值对比”的完整流程。最终模型必须替换输入维度和输出头，并把“输出相同”定义为密钥/解密信号的数值等价，而不是分类概率相同。

### 0.2 先明确“成熟”的含义

本文没有把“网上出现过”当作成熟。纳入最终比较的路线至少同时满足以下条件：

- 有 AMD/Xilinx 或主流机器学习硬件社区维护的官方仓库。
- 有近年的正式版本或持续维护记录。
- 有从训练框架到 FPGA 硬件或 `.xmodel`/bitstream 的完整说明。
- 对量化、综合、板端部署和常见错误有可追踪文档。
- 能容纳本项目的 Conv2d、MaxPool2d、ReLU、Flatten、Linear 结构。

最终采用的官方依据包括：

- FINN v0.10.1：<https://github.com/Xilinx/finn>
- FINN Getting Started：<https://finn.readthedocs.io/en/latest/getting_started.html>
- FINN CNN 端到端示例：<https://github.com/Xilinx/finn/blob/main/notebooks/end2end_example/bnn-pynq/cnv_end2end_example.ipynb>
- Brevitas：<https://github.com/Xilinx/brevitas>
- hls4ml v1.3.0：<https://github.com/fastmachinelearning/hls4ml>
- hls4ml 文档：<https://fastmachinelearning.org/hls4ml/>
- hls4ml PyTorch 前端：<https://github.com/fastmachinelearning/hls4ml/blob/main/docs/frontend/pytorch.rst>
- hls4ml VivadoAccelerator：<https://github.com/fastmachinelearning/hls4ml/blob/main/docs/backend/accelerator.rst>
- Vitis AI v5.0：<https://github.com/Xilinx/Vitis-AI>
- Vitis AI 5.0 文档：<https://xilinx.github.io/Vitis-AI/5.0/>
- Vitis AI Zynq MPSoC Quickstart：<https://xilinx.github.io/Vitis-AI/5.0/html/docs/quickstart/mpsoc.html>
- Vitis AI PyTorch 量化示例：<https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/example/resnet18_quant.py>

### 0.3 筛选掉的其他路线

下面这些路线并非不能用，而是不适合作为本教程的“前三名”：

- **手写 Verilog/SystemVerilog**：控制和时序最自由，但卷积、行缓存、池化、AXI、定点缩放、全连接调度都要自己实现，开发与验证成本远高于本项目收益。
- **手写 Vitis HLS C++**：比手写 RTL 高效，是成熟 FPGA 技术，但需要手工实现逐层循环、接口、量化和小数点管理；作为学习项目很好，作为“从 PyTorch 移植”的通用路线不如 hls4ml。
- **Vitis AI Library 自带的完整应用库**：Vitis AI 5.0 已明确说明该版本未启用 `vitis_ai_library`；若必须使用，要选择兼容的旧版本。
- **DPU-PYNQ**：方便，但仓库已归档，依赖 PYNQ 3.0 和 Vitis AI 2.5.0 一类的旧组合。它可以作为旧环境复现资料，不应把旧版本当作 5.0 的默认部署方式。
- **TensorFlow Lite Micro、CMSIS-NN 等 MCU 路线**：擅长 MCU，不是通用 FPGA 部署栈。
- **旧 SDAccel/OpenCL 路线**：历史资料多，但已不属于当前首选 AMD 开发流程。

---

## 1. 验证模型与部署约束

当前 `DigitCNN` 不是已经验证过的混沌解密器，它承担的是“部署试验台”角色。真正移植时需要做以下映射：

- 若最终模型仍是二维 CNN，现有卷积层、池化层和 AXI 流式部署方法基本可以复用，只替换输入尺寸和最后一层。
- 若最终模型使用一维卷积处理混沌时间序列，需要确认 hls4ml/FINN 对 `Conv1d` 的支持，并同时记录窗口长度、历史缓存和序列边界。
- 若最终模型包含 GRU、LSTM、注意力或混沌方程迭代，DPU 通常不能直接编译；应把神经网络部分和专用递推/算术部分拆成不同硬件模块。
- 神经网络若只产生密钥流，应明确密钥的分组长度、字节序、同步标记和重置规则；量化误差可能导致整段视频持续错位，而不只是单次分类错误。

### 1.1 网络结构

`model.py` 中的 `DigitCNN` 结构如下。下面的索引是 `nn.Sequential` 中的实际索引，后续手动复制权重时会用到。

| 索引 | 层 | 输入形状 | 输出形状 | 参数量 | FPGA 处理意见 |
| --- | --- | --- | --- | ---: | --- |
| `features.0` | Conv2d 3×3，pad=1，1→32 | `1×28×28` | `32×28×28` | 288 + 32 | 保留 |
| `features.1` | ReLU | `32×28×28` | `32×28×28` | 0 | 保留，在 HLS/DPU 中可作为激活 |
| `features.2` | Conv2d 3×3，pad=1，32→32 | `32×28×28` | `32×28×28` | 9,216 + 32 | 保留 |
| `features.3` | ReLU | `32×28×28` | `32×28×28` | 0 | 保留 |
| `features.4` | MaxPool2d 2×2 | `32×28×28` | `32×14×14` | 0 | 保留 |
| `features.5` | Conv2d 3×3，pad=1，32→64 | `32×14×14` | `64×14×14` | 18,432 + 64 | 保留 |
| `features.6` | ReLU | `64×14×14` | `64×14×14` | 0 | 保留 |
| `features.7` | Conv2d 3×3，pad=1，64→64 | `64×14×14` | `64×14×14` | 36,864 + 64 | 保留 |
| `features.8` | ReLU | `64×14×14` | `64×14×14` | 0 | 保留 |
| `features.9` | MaxPool2d 2×2 | `64×14×14` | `64×7×7` | 0 | 保留 |
| `features.10` | Dropout2d | `64×7×7` | `64×7×7` | 0 | 推理时删除 |
| `classifier.0` | Flatten | `64×7×7` | `3,136` | 0 | 保留为 reshape |
| `classifier.1` | Linear 3136→128 | `3,136` | `128` | 401,408 + 128 | 部署层中最大的权重来源 |
| `classifier.2` | ReLU | `128` | `128` | 0 | 保留 |
| `classifier.3` | Dropout | `128` | `128` | 0 | 推理时删除 |
| `classifier.4` | Linear 128→10 | `128` | `10` | 1,280 + 10 | 保留 |

总参数量为 **467,818**。其中约 **86%** 的参数集中在 `3136→128` 全连接层，所以优化时不能只关注前四个卷积层。若目标板 BRAM 紧张，全连接层通常比卷积层更需要定点压缩或外存流式读取。

前向推理约包含 **13,272,832 次乘法累加（MAC）**，不计池化、比较和地址计算。对大型 DPU 来说运算量很小；对资源受限器件，若把所有乘加全部展开仍可能占用过多 DSP 和逻辑资源，因此必须根据所选器件使用复用因子、低比特量化或流式调度。

### 1.2 从分类准确率转向解密指标

仓库中的 `outputs/metrics.json` 当前记录：

- 测试准确率：**99.42%**
- 最佳验证轮次：第 27 轮
- 训练设备：CUDA

当前 `DigitCNN` 的 99.42% 仅用于确认验证模型和部署流程可用。对于最终解密模型，应分别记录：

1. 原始 float32 PyTorch 输出与参考解密结果。
2. 定点或 INT8 模拟器输出的误码率（BER）、符号错误率或数值误差。
3. HLS/RTL 联合仿真和真实板端输出。
4. 解密帧的 PSNR、SSIM、颜色/像素误差以及是否逐位一致。
5. 连续序列的丢包、重复、错位、帧间隔抖动和持续吞吐。

验证模型阶段仍可保留 MNIST 测试集准确率不低于 98% 的门槛；最终项目不能沿用该数字代替解密验收。BER、PSNR/SSIM 和逐位一致性阈值应由混沌映射、纠错能力和视频编码容忍度共同决定，并在上板前固定测试协议。

### 1.3 必须固定的输入契约

下面先给出当前 MNIST 验证模型的输入约定。最终混沌信号应改为记录采样率、每帧样本数、实部/虚部或幅值格式、量化范围、字节序、序列号和同步标记；图像均值与标准差不能直接套用到信号数据。

当前 PyTorch 模型的训练输入是：

```text
灰度值 0..255
→ ToTensor 后除以 255，得到 0..1
→ 减 0.1307、除以 0.3081
→ NCHW: 1×28×28
```

对应数学表达式：

```text
x_normalized = (x_uint8 / 255.0 - 0.1307) / 0.3081
```

其数值范围约为 `-0.424..2.821`，不是常见的 `-1..1`。这是 FPGA 输入量化最容易出错的地方之一。必须回答并写进部署配置的三个问题是：

- PC 端送给 FPGA 的是 `uint8 0..255`、`float 0..1`，还是已经标准化后的 `float -0.424..2.821`？
- FPGA 内部是否包含除法和减法？
- 如果量化器只接受 `-1..1`，是否在训练阶段重新设计了输入范围？

本文的统一约定是：

- hls4ml 和 Vitis AI 默认使用现有的标准化输入，PS 端先完成归一化。
- FINN 路线建议重新训练量化模型，输入直接使用 `0..1`，避免把负范围和较大的正范围硬塞进对称窄位宽。
- 三套方案完成居中 28×28 后得到的**原始 uint8 图像必须完全一致**。

对当前验证模型，`app.py` 的画布是黑底白字，缩放函数应继续使用 `predict.py` 中 `center_on_mnist_canvas` 的等比缩放与居中逻辑。最终项目的测试输入则应来自固定样本的加密信号或加密视频帧，不应再用 MNIST 画布代替真实数据分布。

### 1.4 推理图与训练图分开

Dropout 在 `model.eval()` 时本来就是恒等操作，但某些转换器仍会把 Dropout 节点留在图里。最稳妥的方法是构造一个只用于部署的 `DeployDigitCNN`：

- 保留与训练相同的卷积权重、偏置、全连接权重和偏置。
- 删除两个 Dropout。
- 不加入 BatchNorm，因为当前模型没有 BatchNorm。
- 保留 ReLU、池化和 Flatten。
- 验证阶段不在 FPGA 中实现 Softmax，直接输出 10 个 logits；最终解密模型直接输出所需宽度的恢复信号或密钥张量。

下面是最小部署模型，可直接保存为 `fpga/deploy_model.py`。每一行可执行代码后都有中文注释。

先在项目根目录创建教程使用的脚本目录：

```bash
mkdir -p fpga  # 创建部署模型、测试数据和转换脚本目录
```

```python
import sys  # 导入系统路径工具
from pathlib import Path  # 导入跨平台路径工具

import torch  # 导入 PyTorch 主模块，用于张量计算
from torch import nn  # 导入神经网络模块，并简写为 nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 找到部署脚本所在项目的根目录
if str(PROJECT_ROOT) not in sys.path:  # 判断项目根目录是否已可导入
    sys.path.insert(0, str(PROJECT_ROOT))  # 把项目根目录加入 Python 导入搜索路径

from model import DigitCNN  # 从原项目导入训练时使用的网络结构


class DeployDigitCNN(nn.Module):  # 定义只包含推理必需运算的部署网络
    def __init__(self) -> None:  # 初始化各层
        super().__init__()  # 初始化 PyTorch 模块基类
        self.features = nn.Sequential(  # 建立卷积特征提取部分
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 第一层卷积保持 28×28 尺寸
            nn.ReLU(inplace=True),  # 第一层 ReLU 激活
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 第二层卷积保持 28×28 尺寸
            nn.ReLU(inplace=True),  # 第二层 ReLU 激活
            nn.MaxPool2d(kernel_size=2),  # 第一次池化把尺寸缩小到 14×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 第三层卷积增加通道数
            nn.ReLU(inplace=True),  # 第三层 ReLU 激活
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 第四层卷积保持 14×14 尺寸
            nn.ReLU(inplace=True),  # 第四层 ReLU 激活
            nn.MaxPool2d(kernel_size=2),  # 第二次池化把尺寸缩小到 7×7
        )  # 结束特征提取部分
        self.classifier = nn.Sequential(  # 建立分类器部分
            nn.Flatten(),  # 把 64×7×7 展平为 3136 个元素
            nn.Linear(64 * 7 * 7, 128),  # 第一层全连接把特征压缩到 128 维
            nn.ReLU(inplace=True),  # 全连接后的 ReLU 激活
            nn.Linear(128, 10),  # 输出 10 个未归一化类别分数
        )  # 结束分类器部分

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义部署网络前向计算
        x = self.features(x)  # 提取输入图像特征
        x = self.classifier(x)  # 把特征映射为 10 个 logits
        return x  # 返回 logits，Softmax 留到 CPU 端


def load_deploy_model(checkpoint_path: str, device: str = "cpu") -> DeployDigitCNN:  # 从现有检查点构造部署模型
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)  # 安全加载训练检查点
    source = DigitCNN()  # 创建原训练网络以读取完整参数
    source.load_state_dict(checkpoint["model_state_dict"])  # 把训练权重加载到原网络
    target = DeployDigitCNN()  # 创建不含 Dropout 的部署网络
    target.features[0].load_state_dict(source.features[0].state_dict())  # 复制第一层卷积权重和偏置
    target.features[2].load_state_dict(source.features[2].state_dict())  # 复制第二层卷积权重和偏置
    target.features[5].load_state_dict(source.features[5].state_dict())  # 复制第三层卷积权重和偏置
    target.features[7].load_state_dict(source.features[7].state_dict())  # 复制第四层卷积权重和偏置
    target.classifier[1].load_state_dict(source.classifier[1].state_dict())  # 复制第一层全连接参数
    target.classifier[3].load_state_dict(source.classifier[4].state_dict())  # 复制输出层全连接参数
    target.eval()  # 切换到推理模式
    return target  # 返回可直接转换的部署网络
```

注意：不要用“新建模型后只调用 `load_state_dict`”的方式加载原检查点，因为删除 Dropout 后两个 `Sequential` 的索引会变化，完整状态字典无法直接匹配。上面的逐层复制是显式且安全的。

### 1.5 保存一份共同测试集

三套工具都应使用完全相同的原始图像、标签和标准化张量。建议只生成一次固定测试包。下面代码可保存为 `fpga/make_golden_set.py`；默认取前 2,048 张测试图片，最后验收仍应跑完整 10,000 张。

该脚本是工具链验证包。最终项目应另建 `golden_signal.npz`，保存加密输入、参考密钥/恢复信号、帧号、CRC 或哈希、参考视频帧以及 PyTorch 输出；不要用 MNIST 指标代替混沌信号的 BER 和视频恢复质量。

```python
import argparse  # 导入命令行参数工具
from pathlib import Path  # 导入跨平台路径工具

import numpy as np  # 导入 NumPy 用于保存数组
import torch  # 导入 PyTorch 用于批处理和模型计算
from torch import nn  # 导入神经网络模块用于设备包装
from torch.utils.data import DataLoader, Subset  # 导入数据子集和加载器
from torchvision import datasets, transforms  # 导入 MNIST 和图像变换

from model import DigitCNN  # 导入原始网络结构


MEAN = (0.1307,)  # 定义训练使用的一通道均值
STD = (0.3081,)  # 定义训练使用的一通道标准差


def parse_args() -> argparse.Namespace:  # 定义参数解析函数
    parser = argparse.ArgumentParser(description="生成 FPGA 与 PyTorch 共用的测试数据")  # 创建参数解析器
    parser.add_argument("--model", type=Path, default=Path("outputs/best_model.pth"))  # 指定模型检查点路径
    parser.add_argument("--data-dir", type=Path, default=Path("data"))  # 指定 MNIST 数据目录
    parser.add_argument("--output", type=Path, default=Path("fpga/golden_mnist.npz"))  # 指定输出文件
    parser.add_argument("--count", type=int, default=2048)  # 指定快速验证样本数量
    args = parser.parse_args()  # 解析命令行输入
    args.output.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    return args  # 返回参数


def main() -> None:  # 定义主流程
    args = parse_args()  # 读取参数
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])  # 建立与训练一致的变换
    raw_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transforms.ToTensor())  # 读取 0..1 测试图像以便可靠还原 uint8
    norm_dataset = datasets.MNIST(root=args.data_dir, train=False, download=False, transform=transform)  # 读取标准化测试图像
    indices = list(range(min(args.count, len(raw_dataset))))  # 选择固定且可复现的样本编号
    loader = DataLoader(Subset(norm_dataset, indices), batch_size=256, shuffle=False)  # 创建按顺序读取的批加载器
    x_normalized = torch.cat([images for images, _ in loader], dim=0)  # 拼接全部标准化张量
    t_loader = DataLoader(Subset(raw_dataset, indices), batch_size=256, shuffle=False)  # 创建原始图像加载器
    x_uint8 = torch.cat([(images * 255.0).round().to(torch.uint8) for images, _ in t_loader], dim=0)  # 把 0..1 图像精确还原为 0..255 灰度图
    labels = torch.tensor([raw_dataset[index][1] for index in indices], dtype=torch.long)  # 读取固定样本标签
    model = DigitCNN()  # 创建原始模型
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)  # 加载检查点
    model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型参数
    model.eval()  # 关闭 Dropout
    logits = []  # 初始化 logits 列表
    with torch.inference_mode():  # 禁止梯度记录以降低内存
        for start in range(0, len(x_normalized), 256):  # 按批次执行推理
            logits.append(model(x_normalized[start:start + 256]))  # 计算当前批次 logits
    y_logits = torch.cat(logits, dim=0).numpy().astype(np.float32)  # 合并并转换为 float32
    accuracy = float((y_logits.argmax(axis=1) == labels.numpy()).mean())  # 计算参考准确率
    np.savez_compressed(  # 保存压缩数组文件
        args.output,  # 指定输出路径
        x_uint8=x_uint8.numpy(),  # 保存 FPGA 预处理前图像
        x_nchw=x_normalized.numpy().astype(np.float32),  # 保存 PyTorch NCHW 输入
        labels=labels.numpy().astype(np.int64),  # 保存标签
        y_logits=y_logits,  # 保存参考 logits
    )  # 结束文件保存
    print(f"已保存 {len(labels)} 个样本，参考准确率 {accuracy:.2%}")  # 输出样本数和准确率


if __name__ == "__main__":  # 只有直接执行脚本时才运行
    main()  # 启动主流程
```

生成后立刻检查数组形状：

```python
import numpy as np  # 导入数组工具
data = np.load("fpga/golden_mnist.npz")  # 加载测试包
print(data["x_uint8"].shape)  # 应输出 (N, 1, 28, 28)
print(data["x_nchw"].shape)  # 应输出 (N, 1, 28, 28)
print(data["labels"].shape)  # 应输出 (N,)
print(data["y_logits"].shape)  # 应输出 (N, 10)
```

---

## 2. 路线一：hls4ml 生成 HLS 神经网络 IP

### 2.1 原理与适用性

hls4ml 会把经过 `torch.fx` 符号追踪的 PyTorch 计算图转换成 HLS C++，再由 Vivado HLS 综合成 IP。IP 通过 AXI Stream 接收信号并返回输出；它可以集成到自定义 Vivado Block Design，也可以在使用官方加速器适配时封装成 overlay。

本项目的四层卷积、ReLU、MaxPool、Flatten 和 Dense 都在 hls4ml 的常见支持范围内。官方 PyTorch 前端目前仍不如 Keras 前端覆盖全面，因此必须把自定义 Python 控制流控制在最低限度。现有模型只使用 `nn.Module`，适合改造成可追踪模型。

最终解密模型若是纯 CNN，可直接沿用本节流程；若含一维卷积，应检查 `Conv1d`、窗口边界和流式缓存支持。混沌状态递推、帧同步和可变长度控制不宜藏进 `forward`，更适合放到 HLS 外的 PL 状态机或 PS 软件。

适合以下情况：

- 主要目的是学习从 PyTorch 到 HLS 的流程。
- 愿意先在工具链仿真中确定模型，再根据最终器件完成 AXI/DMA 集成。
- 可以接受固定点数量化，并能根据资源和时序调整并行度。
- 暂时没有精力重写一套 Brevitas 训练代码。

### 2.2 版本、许可证与系统要求

建议锁定 hls4ml 1.x，例如：

```bash
python -m pip install "hls4ml[profiling]>=1.3,<2"  # 安装 hls4ml 及数值分析扩展
```

还需要：

- 安装完整 AMD Vivado。只安装 Python 包不能完成综合和 bitstream 生成。
- 在 Linux x86-64 环境运行，Vivado 版本必须与所安装 hls4ml 的发布说明匹配。
- 若使用官方加速器封装，需要对应平台的驱动和运行时；自定义板卡则需自行提供 DMA、地址映射和主机/PS 驱动。
- 资源受限器件若一开始就把 `ReuseFactor` 设为 1，综合很可能因为并行乘加过多而资源爆炸。

先检查工具链：

```bash
source /opt/Xilinx/Vivado/2024.2/settings64.sh  # 加载当前环境的 Vivado 命令和许可证配置
vivado -version  # 确认 Vivado 可执行并能找到许可证
python -c "import hls4ml; print(hls4ml.__version__)"  # 确认 hls4ml 版本
```

上面的 `/opt/Xilinx/Vivado/2024.2` 只是示例路径，应改成本机实际安装位置。

### 2.3 转换前的结构检查

hls4ml 的 PyTorch 前端依赖 `torch.fx`。转换前应满足：

- 模型能在 `model.eval()` 下正常前向。
- 不在 `forward` 中读取 Python 列表长度后做动态分支。
- 不使用自定义 Python 循环隐藏计算。
- 不使用未知自定义算子。
- Dropout 已从部署模型中删除。
- 验证模型输入固定为 `1×28×28`、输出固定为 10 类；最终模型改为固定信号窗口和密钥/恢复信号宽度。
- 模型内部保持 NCHW；hls4ml 内部是 NHWC 风格，转换和测试时要注意布局。

官方文档特别指出：`io_parallel` 会自动为输入加入转置节点，但 `io_stream` 不会，使用 `io_stream` 时必须由用户准备正确布局。本项目建议综合时使用 `io_stream`，数值对照时明确送入 NHWC。

### 2.4 生成部署模型

先保存第 1.4 节的 `fpga/deploy_model.py`。下面的转换脚本可保存为 `fpga/hls4ml_build.py`，并放在项目根目录执行，或者把项目根目录加入 `PYTHONPATH`。

```python
import argparse  # 导入命令行参数工具
from pathlib import Path  # 导入路径工具

import hls4ml  # 导入 hls4ml 主包
import numpy as np  # 导入 NumPy 进行布局转换和精度统计
import torch  # 导入 PyTorch 读取参考输入

from deploy_model import load_deploy_model  # 导入第 1.4 节的部署模型加载函数


def parse_args() -> argparse.Namespace:  # 定义参数解析
    parser = argparse.ArgumentParser(description="把 DigitCNN 转换为 hls4ml 工程")  # 创建参数解析器
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pth"))  # 指定训练检查点
    parser.add_argument("--golden", type=Path, default=Path("fpga/golden_mnist.npz"))  # 指定共同测试包
    parser.add_argument("--output", type=Path, default=Path("fpga/hls4ml_mnist"))  # 指定 hls4ml 输出目录
    parser.add_argument("--backend", choices=["Vivado", "VivadoAccelerator"], default="Vivado")  # 选择通用 HLS IP 或官方加速器封装
    parser.add_argument("--board", default=None)  # 可选指定 hls4ml 官方支持的板卡标识
    parser.add_argument("--reuse-factor", type=int, default=64)  # 指定优先使用的小器件并行复用因子
    args = parser.parse_args()  # 解析输入
    if args.backend == "VivadoAccelerator" and args.board is None:  # 检查官方加速器模式是否提供目标
        parser.error("VivadoAccelerator 模式必须提供 --board；自定义 FPGA 请使用 Vivado 后端生成 IP")  # 给出目标中立提示
    args.output.parent.mkdir(parents=True, exist_ok=True)  # 确保输出父目录存在
    return args  # 返回参数对象


def make_config(model: torch.nn.Module, reuse_factor: int) -> dict:  # 建立固定点与资源配置
    config = hls4ml.utils.config_from_pytorch_model(  # 从 PyTorch 模型推导基础配置
        model,  # 传入可追踪网络
        (1, 28, 28),  # 声明去批次后的输入尺寸
        granularity="type",  # 按层类型生成 Conv2D、Dense 等公共配置
        default_precision="ap_fixed<16,6>",  # 先用较稳妥的 16 位默认精度
        backend="Vivado",  # 使用 Vivado HLS 配置体系
    )  # 完成基础配置生成
    config["Model"]["Strategy"] = "Resource"  # 优先节省逻辑资源
    config["Model"]["ReuseFactor"] = reuse_factor  # 让乘加单元在多个输出之间复用
    config["Model"]["Precision"]["default"] = "ap_fixed<16,6>"  # 指定模型默认数据精度
    precision_config = {  # 定义卷积和全连接共用的定点精度
        "weight": "ap_fixed<8,2>",  # 权重使用 8 位有符号定点
        "bias": "ap_fixed<16,6>",  # 偏置保留较宽范围
        "result": "ap_fixed<16,6>",  # 卷积结果保留较宽范围
        "accum": "ap_fixed<28,10>",  # 累加器额外留出整数位防止溢出
    }  # 结束定点精度配置
    if "Conv2D" in config["LayerType"]:  # 检查当前 hls4ml 版本是否使用 Conv2D 类型名
        config["LayerType"]["Conv2D"]["Precision"] = precision_config.copy()  # 覆盖卷积层精度
    if "Dense" in config["LayerType"]:  # 检查当前 hls4ml 版本是否使用 Dense 类型名
        config["LayerType"]["Dense"]["Precision"] = precision_config.copy()  # 覆盖全连接层精度
    return config  # 返回可修改的配置字典


def main() -> None:  # 定义主流程
    args = parse_args()  # 读取参数
    model = load_deploy_model(str(args.checkpoint), "cpu")  # 加载无 Dropout 的部署模型
    config = make_config(model, args.reuse_factor)  # 生成定点与资源策略
    print("hls4ml 层类型：", list(config["LayerType"].keys()))  # 打印配置中实际存在的层类型
    backend_options = {"board": args.board} if args.backend == "VivadoAccelerator" else {}  # 只在加速器后端传入板卡名
    hls_model = hls4ml.converters.convert_from_pytorch_model(  # 把 PyTorch 图转换为 HLS 模型
        model,  # 传入部署模型
        hls_config=config,  # 传入精度和复用配置
        output_dir=str(args.output),  # 指定生成工程目录
        backend=args.backend,  # 选择 Vivado 或 VivadoAccelerator
        io_type="io_stream",  # 使用流式输入输出以降低全展开并行资源
        **backend_options,  # 仅给 VivadoAccelerator 添加 board 参数
    )  # 完成转换
    hls_model.write()  # 把 HLS C++、Tcl 和工程文件写入输出目录
    golden = np.load(args.golden)  # 加载共同测试数据
    x_nchw = golden["x_nchw"].astype(np.float32)  # 取出 PyTorch NCHW 输入
    labels = golden["labels"].astype(np.int64)  # 取出标签
    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1))  # io_stream 使用 NHWC，手动完成通道布局转换
    hls_model.compile()  # 编译 C++ 仿真模型，不生成 bitstream
    y_hls = np.asarray(hls_model.predict(x_nhwc))  # 用 HLS C 仿真计算输出
    y_ref = golden["y_logits"].astype(np.float32)  # 读取 PyTorch 参考 logits
    accuracy_ref = float((y_ref.argmax(axis=1) == labels).mean())  # 计算 PyTorch 固定样本准确率
    accuracy_hls = float((y_hls.argmax(axis=1) == labels).mean())  # 计算 HLS 固定样本准确率
    agreement = float((y_hls.argmax(axis=1) == y_ref.argmax(axis=1)).mean())  # 计算两边预测一致率
    max_abs_error = float(np.max(np.abs(y_hls - y_ref)))  # 记录最大 logit 绝对误差
    print(f"PyTorch={accuracy_ref:.2%}, HLS={accuracy_hls:.2%}, 一致率={agreement:.2%}, 最大误差={max_abs_error:.4f}")  # 打印数值比对
    if accuracy_hls < 0.98:  # 检查固定点准确率是否跌破目标
        raise SystemExit("固定点准确率低于 98%，请先加宽精度或重新量化，不要急着生成 bitstream")  # 阻止错误模型继续综合


if __name__ == "__main__":  # 只有直接执行脚本时才运行
    main()  # 启动转换流程
```

第一次运行时，建议先以普通 `Vivado` 后端和少量样本确认 HLS 仿真正确，再改用 `VivadoAccelerator` 打包。直接生成 bitstream 前，务必先通过 `compile()` 和 `predict()` 的数值门槛。

### 2.5 精度、资源和布局调优

hls4ml 的优化应先做“数值等价”，再做“资源与时序”。推荐顺序：

1. 先用 `ap_fixed<16,6>` 权重、结果和累加器，确认仿真准确率。
2. 权重降到 `ap_fixed<8,2>`，观察准确率和资源。
3. 激活值可尝试 `ap_fixed<8,3>` 或 QONNX 量化，但必须重新跑固定样本和完整测试集。
4. 增大 `ReuseFactor`，用时间换 DSP/逻辑资源。
5. 如果 `io_parallel` 综合过大，改用 `io_stream` 和 `Resource` 策略。
6. 如果出现严重数值偏差，优先检查 `accum` 整数位、输入范围、权重通道顺序和输出转置。

常见配置作用：

| 参数 | 增大后的效果 | 代价 |
| --- | --- | --- |
| `ReuseFactor` | 每个硬件乘法器服务更多乘加，逻辑/DSP 减少 | 延迟增加 |
| 权重位宽 | 权重存储、DSP 宽度和带宽增加 | 资源和动态功耗增加 |
| 累加器位宽 | 溢出风险降低 | 加法器和寄存器变宽 |
| `io_parallel` | 一次处理更多元素 | 展开循环，资源可能迅速增大 |
| `io_stream` | 按流处理、便于连接 DMA | 需要关心输入布局和吞吐 |
| `Strategy=Resource` | 优先降低面积 | 延迟和流水效率可能下降 |

不要在没有综合报告时声称具体 DSP、LUT、BRAM 数量。不同 hls4ml 版本、Vivado 版本、精度、`ReuseFactor` 和板卡封装都会改变结果。

### 2.6 可选：生成官方加速器封装

普通 HLS IP 是更通用的路线，适合集成到你最终选定的 Vivado Block Design。只有目标恰好被当前 hls4ml 的 `VivadoAccelerator` 后端支持时，才使用下面的可选封装；把 `BOARD_ID` 换成该版本文档实际列出的标识：

```python
import hls4ml  # 导入 hls4ml

from deploy_model import load_deploy_model  # 导入部署模型加载函数


BOARD_ID = "<当前hls4ml版本支持的板卡标识>"  # 占位符，必须按实际版本和支持列表替换
model = load_deploy_model("outputs/best_model.pth", "cpu")  # 加载无 Dropout 模型
config = hls4ml.utils.config_from_pytorch_model(  # 从模型生成基础 hls4ml 配置
    model,  # 传入 PyTorch 模型
    (1, 28, 28),  # 指定单张输入尺寸
    granularity="name",  # 允许按名称调整层
    default_precision="ap_fixed<16,6>",  # 使用稳妥的默认精度
    backend="Vivado",  # 先生成 Vivado HLS 配置
)  # 完成配置生成
config["Model"]["Strategy"] = "Resource"  # 资源优先
config["Model"]["ReuseFactor"] = 64  # 对小器件使用较大的复用因子
hls_model = hls4ml.converters.convert_from_pytorch_model(  # 转换为官方加速器后端工程
    model,  # 传入部署模型
    hls_config=config,  # 传入配置
    output_dir="fpga/hls4ml_prj_accel",  # 指定目标无关的输出目录
    backend="VivadoAccelerator",  # 使用 VivadoAccelerator 后端
    board=BOARD_ID,  # 使用当前 hls4ml 版本明确支持的板卡标识
    io_type="io_stream",  # 使用 AXI Stream 风格 I/O
)  # 完成转换
hls_model.build(bitfile=True)  # 运行综合与实现并生成 bit 文件
```

官方 `VivadoAccelerator` 后端只接受其当前版本明确列出的板卡标识，而且列表会随版本变化。若目标板不在列表中，继续使用普通 HLS IP，把 AXI Stream 接口接入自定义 DMA 和地址映射；不要借用相近板卡名称。

### 2.7 可选的官方封装打包示例

如果目标属于官方 `VivadoAccelerator` 支持范围，可按官方示例打包 bit、硬件 handoff 和 Python 驱动。下面目录名只是示例，应由实际生成工程替换：

```bash
mkdir -p package  # 创建只包含部署文件的目录
cp fpga/hls4ml_prj_accel/myproject_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper.bit package/hls4ml_nn.bit  # 复制实际生成的 FPGA bit 文件
cp fpga/hls4ml_prj_accel/myproject_vivado_accelerator/project_1.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh package/hls4ml_nn.hwh  # 复制实际生成的硬件 handoff 描述
cp fpga/hls4ml_prj_accel/axi_stream_driver.py package/  # 复制该后端实际生成的 DMA 驱动
tar -czvf package.tar.gz -C package .  # 压缩成可上传到板卡的部署包
```

将 `package.tar.gz` 复制到目标平台，解压后执行。下面的 `NeuralNetworkOverlay` 驱动仅在官方加速器后端生成该 Python 适配层时存在：

```python
from axi_stream_driver import NeuralNetworkOverlay  # 导入 hls4ml 官方后端生成的 overlay 驱动


nn = NeuralNetworkOverlay("hls4ml_nn.bit", (1, 28, 28, 1), (1, 10))  # 创建单张图片的输入输出缓冲区
prediction, latency, throughput = nn.predict(input_one, profile=True)  # 发送一张 NHWC 输入并测量延迟
print(prediction.argmax(axis=1))  # 在 PS 端取得最高分数字
```

这里的 `input_one` 必须是 shape 为 `(1, 28, 28, 1)` 的 NHWC float 数组。若驱动版本或生成工程改写后调用接口发生变化，应以生成目录中的 `axi_stream_driver.py` 为准。最终项目应把多帧加密数据组织成连续批次，并验证长时间运行时 FIFO 和 DMA 是否稳定。

### 2.8 hls4ml 调试清单

- **转换时报 `torch.fx` 追踪失败**：把动态 Python 分支移到模型外，避免在 `forward` 中读取可变长度列表。
- **综合资源爆炸**：先增大 `ReuseFactor`，再改用 `io_stream`/`Resource`；不要在第一次尝试就全并行。
- **输出类别顺序看似随机**：核对 NCHW→NHWC、输出是否被转置，以及权重是否按层名正确配置。
- **仿真全为同一数字**：检查输入是否存在 `0..255` 与 `0..1` 混用，或标准化范围是否被裁掉。
- **HLS 准确率接近随机**：对单张参考图逐层比较中间结果，优先查输入布局和全连接 Flatten 顺序。
- **板端 DMA 超时**：检查输入字节数、对齐要求、overlay 是否加载成功、DMA 与 IP 是否都处于预期时钟域。
- **HLS 仿真通过但板上失败**：比对 `hls_model.predict()` 与板端对同一 `x_nhwc` 的结果，避免测试数据在传输时被重复归一化。

---

## 3. 路线二：FINN + Brevitas 数据流加速器

### 3.1 原理与适用性

FINN 会把量化神经网络编译成以 AXI Stream 连接的数据流硬件。卷积和矩阵向量乘法会被映射成多阈值激活、矩阵向量激活单元、滑动窗口、池化/FIFO 等硬件层。它特别关注低比特权重和激活，例如 1–8 bit，而不是普通的 float32 HLS 网络。

对本项目而言，FINN 的优势是：

- 该 CNN 尺寸不大，适合做成流式加速器。
- 4W4A 或 8W8A 有机会明显减少 BRAM/DSP 压力。
- FINN 的 `PE` 和 `SIMD` 折叠参数可以直接在延迟与资源之间权衡。
- 官方有从 Brevitas CNN 到 PYNQ 部署的端到端 notebook，和本项目的四卷积加全连接结构较接近。

对最终解密项目，FINN 更适合固定的 CNN 密钥生成器或固定长度信号恢复网络。若解密正确性依赖长时间递归状态、浮点混沌方程或动态序列长度，不能假设 FINN 会自动覆盖这些逻辑；应把可量化的 CNN 部分交给 FINN，其余部分放在 PL/PS 中并通过明确的握手协议连接。

FINN 的代价是：

- 不能直接把现有 float `best_model.pth` 丢进编译器。
- 需要 Brevitas 量化训练/微调，再导出 QONNX。
- 网络拓扑要经过 FINN 的硬件层推断，Flatten、Linears、reshape 和输出节点需要按官方模式整理。
- 完整工具链主要通过 Docker 提供，且仍需要匹配的 Vivado/Vitis。

### 3.2 环境与版本

本文编写时 FINN 的正式发布版本是 **v0.10.1**，官方 README 说明编译器当前仅支持 Docker 方式执行。Brevitas 版本需与 FINN 要求匹配；不要在不同容器中随意混装 `brevitas`、`qonnx`、`finn` 和 `onnx` 版本。

建议先安装 Docker，再准备 AMD 工具目录。以下命令在每个新终端执行：

```bash
git clone https://github.com/Xilinx/finn.git  # 下载 FINN 官方仓库
cd finn  # 进入 FINN 目录
git checkout v0.10.1  # 切换到固定正式版本便于复现
export FINN_XILINX_PATH=/tools/Xilinx  # 指定 AMD 工具安装根目录
export FINN_XILINX_VERSION=2024.2  # 指定与 FINN 文档匹配的工具版本
./run-docker.sh quicktest verify  # 运行非 Vivado 的快速自检
```

`FINN_XILINX_PATH` 和 `FINN_XILINX_VERSION` 必须改成实际路径与版本。FINN 官方当前 Getting Started 以 2024.2 为示例；如果你的 FINN 分支要求其他版本，以该分支文档为准。

若只想先验证 Brevitas 导出的 ONNX 语义，可以在普通 Python 环境中安装与 FINN 版本对应的 Brevitas/QONNX，但真正构建 bitstream 时应在 FINN Docker 中完成，避免主机与容器版本不一致。

### 3.3 为 FINN 重新训练量化 CNN

为了让 FINN 的输入量化更自然，FINN 路线建议使用原始 `0..1` 输入重新训练，而不是沿用现有 `(x-0.1307)/0.3081`。网络结构保持本项目拓扑，只把普通层替换为 Brevitas 量化层，并删除 Dropout。

下面代码可保存为 `fpga/finn_brevitas.py`。它依赖 Brevitas 官方示例中的 `CommonWeightQuant` 和 `CommonActQuant`：

```python
import argparse  # 导入命令行参数工具
from pathlib import Path  # 导入路径工具

import copy  # 导入深复制工具用于保存最佳量化权重
import torch  # 导入 PyTorch
from brevitas.export import export_qonnx  # 导入 Brevitas 的 QONNX 导出函数
from brevitas.nn import QuantConv2d  # 导入量化二维卷积
from brevitas.nn import QuantIdentity  # 导入量化恒等层用于激活量化
from brevitas.nn import QuantLinear  # 导入量化全连接层
from brevitas.nn import QuantReLU  # 导入量化 ReLU
from brevitas_examples.bnn_pynq.models.common import CommonActQuant  # 导入官方 FINN 常用激活量化配置
from brevitas_examples.bnn_pynq.models.common import CommonWeightQuant  # 导入官方 FINN 常用权重量化配置
from torch import nn  # 导入基础神经网络模块
from torch.optim import Adam  # 导入 Adam 优化器
from torch.utils.data import DataLoader, random_split  # 导入数据加载器和训练集随机划分
from torchvision import datasets, transforms  # 导入 MNIST 和图像变换


class BrevitasDigitCNN(nn.Module):  # 定义 FINN 路线专用量化网络
    def __init__(self, weight_bits: int = 4, act_bits: int = 4) -> None:  # 初始化量化位宽
        super().__init__()  # 初始化父类
        self.input_quant = QuantIdentity(  # 创建输入量化层
            act_quant=CommonActQuant,  # 使用 FINN 示例常用的激活量化配置
            bit_width=8,  # 输入使用 8 位
            min_val=0.0,  # 输入下限为零
            max_val=1.0,  # 输入上限为一
        )  # 完成输入量化层
        self.features = nn.Sequential(  # 创建量化卷积特征提取部分
            QuantConv2d(1, 32, 3, padding=1, bias=True, weight_quant=CommonWeightQuant, weight_bit_width=weight_bits),  # 第一层量化卷积
            QuantReLU(act_quant=CommonActQuant, bit_width=act_bits),  # 第一层量化 ReLU
            QuantConv2d(32, 32, 3, padding=1, bias=True, weight_quant=CommonWeightQuant, weight_bit_width=weight_bits),  # 第二层量化卷积
            QuantReLU(act_quant=CommonActQuant, bit_width=act_bits),  # 第二层量化 ReLU
            nn.MaxPool2d(2),  # 第一次最大池化
            QuantConv2d(32, 64, 3, padding=1, bias=True, weight_quant=CommonWeightQuant, weight_bit_width=weight_bits),  # 第三层量化卷积
            QuantReLU(act_quant=CommonActQuant, bit_width=act_bits),  # 第三层量化 ReLU
            QuantConv2d(64, 64, 3, padding=1, bias=True, weight_quant=CommonWeightQuant, weight_bit_width=weight_bits),  # 第四层量化卷积
            QuantReLU(act_quant=CommonActQuant, bit_width=act_bits),  # 第四层量化 ReLU
            nn.MaxPool2d(2),  # 第二次最大池化
        )  # 结束特征提取部分
        self.classifier = nn.Sequential(  # 创建量化分类器
            nn.Flatten(),  # 把 64×7×7 展平
            QuantLinear(64 * 7 * 7, 128, bias=True, weight_quant=CommonWeightQuant, weight_bit_width=weight_bits),  # 第一层量化全连接
            QuantReLU(act_quant=CommonActQuant, bit_width=act_bits),  # 第一层全连接后的量化 ReLU
            QuantLinear(128, 10, bias=True, weight_quant=CommonWeightQuant, weight_bit_width=weight_bits),  # 输出 10 个 logits
        )  # 结束分类器部分
        self.output_quant = QuantIdentity(  # 对最终 logits 做量化
            act_quant=CommonActQuant,  # 复用激活量化配置
            bit_width=act_bits,  # 输出使用与激活相同的位宽
        )  # 完成输出量化层

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义量化网络前向计算
        x = self.input_quant(x)  # 把 0..1 输入量化到设定范围
        x = self.features(x)  # 提取量化卷积特征
        x = self.classifier(x)  # 映射为 10 类分数
        return self.output_quant(x)  # 返回量化后的 logits


def evaluate(model: nn.Module, loader: DataLoader) -> float:  # 定义准确率评估函数
    model.eval()  # 切换到评估模式
    correct = 0  # 初始化正确数量
    total = 0  # 初始化总数量
    with torch.inference_mode():  # 禁止梯度计算
        for images, labels in loader:  # 逐批读取数据
            logits = model(images)  # 执行量化模型前向
            correct += int((logits.argmax(dim=1) == labels).sum())  # 累加正确预测数
            total += int(labels.numel())  # 累加样本数
    return correct / total  # 返回准确率


def train_quantized(epochs: int, batch_size: int, learning_rate: float) -> BrevitasDigitCNN:  # 定义量化训练流程
    torch.manual_seed(42)  # 固定随机种子
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())  # 使用 0..1 训练输入
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())  # 使用 0..1 测试输入
    generator = torch.Generator().manual_seed(42)  # 创建可复现的随机划分生成器
    train_subset, validation_subset = random_split(training_data, [54000, 6000], generator=generator)  # 从训练集划出 6000 张验证图片
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)  # 创建打乱训练加载器
    validation_loader = DataLoader(validation_subset, batch_size=512, shuffle=False, num_workers=2)  # 创建验证加载器用于选模
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=2)  # 创建固定顺序测试加载器
    model = BrevitasDigitCNN(weight_bits=4, act_bits=4)  # 从 4W4A 开始尝试
    optimizer = Adam(model.parameters(), lr=learning_rate)  # 创建 Adam 优化器
    criterion = nn.CrossEntropyLoss()  # 创建交叉熵损失
    best_accuracy = 0.0  # 记录最佳验证准确率
    best_state = copy.deepcopy(model.state_dict())  # 初始化最佳权重副本
    for epoch in range(1, epochs + 1):  # 遍历训练轮次
        model.train()  # 切换到训练模式
        for images, labels in train_loader:  # 逐批读取训练数据
            optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度
            loss = criterion(model(images), labels)  # 计算量化网络损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        validation_accuracy = evaluate(model, validation_loader)  # 在当前轮结束后评估验证集
        if validation_accuracy > best_accuracy:  # 判断是否刷新最佳验证准确率
            best_accuracy = validation_accuracy  # 更新最佳验证准确率
            best_state = copy.deepcopy(model.state_dict())  # 保存当前最佳权重
        print(f"epoch={epoch}, validation_accuracy={validation_accuracy:.2%}")  # 打印当前验证准确率
    model.load_state_dict(best_state)  # 恢复验证集表现最好的权重
    test_accuracy = evaluate(model, test_loader)  # 训练结束后只在官方测试集评估一次
    print(f"best_validation={best_accuracy:.2%}, test_accuracy={test_accuracy:.2%}")  # 同时打印验证和测试结果
    if test_accuracy < 0.98:  # 检查量化模型是否满足项目目标
        raise RuntimeError("量化模型未达到 98%，请先调整位宽、轮数或训练参数再导出 FINN 模型")  # 阻止不合格模型进入硬件流程
    return model  # 返回训练好的量化模型


def export_for_finn(model: BrevitasDigitCNN, output: Path) -> None:  # 定义 QONNX 导出函数
    model.eval()  # 切换到评估模式
    sample = torch.rand(1, 1, 28, 28)  # 创建与真实输入同形状的示例
    export_qonnx(model, sample, str(output))  # 导出包含量化算子的 QONNX 模型


def parse_args() -> argparse.Namespace:  # 定义参数解析
    parser = argparse.ArgumentParser(description="训练并导出 FINN Brevitas 模型")  # 创建参数解析器
    parser.add_argument("--epochs", type=int, default=12)  # 指定训练轮数
    parser.add_argument("--batch-size", type=int, default=64)  # 指定批大小
    parser.add_argument("--learning-rate", type=float, default=1e-3)  # 指定学习率
    parser.add_argument("--output", type=Path, default=Path("fpga/finn_mnist/model_qonnx.onnx"))  # 指定输出文件名
    args = parser.parse_args()  # 解析参数
    args.output.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    return args  # 返回参数


def main() -> None:  # 定义主入口
    args = parse_args()  # 读取参数
    model = train_quantized(args.epochs, args.batch_size, args.learning_rate)  # 训练量化模型
    export_for_finn(model, args.output)  # 导出 QONNX
    torch.save(model.state_dict(), args.output.with_suffix(".pth"))  # 同时保存量化模型权重
    print(f"QONNX 已保存到 {args.output}")  # 输出完成提示


if __name__ == "__main__":  # 只有直接执行时才运行
    main()  # 启动训练与导出
```

若 4W4A 达不到 98%，不要直接把它怪罪为 FINN；首先检查：

- 是否真的使用了 `0..1` 输入，且没有重复标准化。
- 量化 ReLU 的输出范围是否覆盖了真实激活值。
- 是否训练轮数太少，或量化参数没有被优化。
- 是否在验证集上观察到明显退化，而不是只看训练集。
- 是否可以尝试 4W8A、8W8A 或逐通道权重量化。

FINN 官方 FAQ 支持 1–8 bit，可做对称权重、逐张量或逐通道缩放，也支持 QAT 和 Brevitas PTQ。对本项目，最稳妥的学习顺序是 `8W8A → 4W8A → 4W4A`。如果 8W8A 都达不到 98%，应先修模型/输入/训练，而不是继续追低比特。

### 3.4 QONNX 清理与 FINN-ONNX 转换

Brevitas 导出的文件不是 FINN 的最终硬件图，需要先做模型清理、形状推断、常量折叠和 QONNX 到 FINN-ONNX 的转换。官方 CNN notebook 的顺序可以照搬。建议在 FINN Docker 中运行下面代码，保存为 `fpga/finn_import.py`：

```python
from pathlib import Path  # 导入路径工具

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN  # 导入 QONNX 到 FINN-ONNX 转换
from qonnx.core.modelwrapper import ModelWrapper  # 导入 QONNX 模型封装器
from qonnx.transformation.fold_constants import FoldConstants  # 导入常量折叠
from qonnx.transformation.general import GiveReadableTensorNames  # 导入可读张量命名
from qonnx.transformation.general import GiveUniqueNodeNames  # 导入唯一节点命名
from qonnx.transformation.general import RemoveStaticGraphInputs  # 导入静态输入移除
from qonnx.transformation.infer_shapes import InferShapes  # 导入形状推断
from qonnx.util.cleanup import cleanup as qonnx_cleanup  # 导入 QONNX 清理函数


source = Path("fpga/finn_mnist/model_qonnx.onnx")  # 指定 Brevitas 导出文件
tidy = Path("fpga/finn_mnist/model.onnx")  # 指定 FINN 输入文件
qonnx_cleanup(str(source), out_file=str(source))  # 清理导出图中的冗余节点
model = ModelWrapper(str(source))  # 读取 QONNX 模型
model = model.transform(ConvertQONNXtoFINN())  # 把量化算子转换为 FINN 可识别形式
model = model.transform(InferShapes())  # 推断所有中间张量形状
model = model.transform(FoldConstants())  # 折叠可提前计算的常量
model = model.transform(GiveUniqueNodeNames())  # 给节点生成唯一名字
model = model.transform(GiveReadableTensorNames())  # 给张量生成便于调试的名字
model = model.transform(RemoveStaticGraphInputs())  # 删除不需要的静态图输入
model.save(str(tidy))  # 保存 FINN-ONNX 模型
print(f"已生成 {tidy}")  # 输出完成路径
```

转换后用 Netron 查看，重点确认：

- 模型输入是 `1×1×28×28` 或经过统一 reshape 后的等价布局。
- 量化层显示为 `Quant`、`MultiThreshold` 等 FINN 认识的形式。
- Flatten 没有变成无法推断的动态形状。
- 卷积的通道顺序、padding、stride 与 PyTorch 一致。
- 最后的 10 类输出仍保持单一输出张量。
- 不能出现训练专用 Dropout 或自定义 Python 层。

FINN 文档明确说明任意 ONNX Zoo 模型不能直接工作，模型必须符合 Brevitas/QONNX/FINN-ONNX 的量化表示。不要把这个警告忽略。

### 3.5 准备构建目录

建议建立下面结构：

```text
fpga/finn_mnist/
├── model.onnx
├── dataflow_build_config.json
├── folding_config.json          # 可选，先由估算结果决定
└── specialize_layers_config.json # 可选，需要特定 HLS 层实现时使用
```

FINN 的 `build_dataflow` 会把构建目录挂载到 Docker 内。下面的目标是占位配置；必须把 `board` 和 `shell_flow_type` 换成所用 FINN 版本能够识别、且与最终器件匹配的值。

```json
{
  "output_dir": "output_digitcnn",
  "target_fps": 1000,
  "mvau_wwidth_max": 10000,
  "synth_clk_period_ns": 10.0,
  "rtlsim_batch_size": 2,
  "board": "<FINN_BOARD_ID>",
  "standalone_thresholds": false,
  "folding_config_file": "folding_config.json",
  "shell_flow_type": "<FINN_SHELL_FLOW>",
  "verify_save_rtlsim_waveforms": false,
  "fifosim_save_waveform": false,
  "debug_fifo": false,
  "verify_steps": [
    "initial_python",
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim"
  ],
  "generate_outputs": [
    "estimate_reports",
    "stitched_ip",
    "rtlsim_performance",
    "pynq_driver",
    "out_of_context_synth",
    "bitfile",
    "deployment_package"
  ]
}
```

其中 `pynq_driver` 是 FINN 模板保留的一项输出；若最终平台不使用该 Python 适配层，可生成 stitched IP 后改用目标平台自己的 DMA/运行时，硬件核心仍是 AXI Stream 数据流。

字段含义：

| 字段 | 作用 | 调整建议 |
| --- | --- | --- |
| `output_dir` | 所有构建结果的输出目录 | 每个实验使用不同名称 |
| `target_fps` | 期望吞吐，影响 folding 候选 | 小项目可先设低值，确认资源后再提高 |
| `mvau_wwidth_max` | 限制矩阵向量单元权重端口宽度 | 过大增加线网和时序压力 |
| `synth_clk_period_ns` | 目标时钟周期，10 ns 约等于 100 MHz | 先保守，再根据时序报告提高频率 |
| `board` | 目标平台标识 | 必须是当前 FINN 版本可识别且与目标器件一致的标识 |
| `shell_flow_type` | 外层硬件 shell 流程 | 按目标平台的处理器系统和接口选择，不要照抄示例 |
| `verify_steps` | 从 Python 到 RTL 的逐层验证 | 学习阶段应保留，速度快后再精简 |
| `generate_outputs` | 最终希望生成的文件 | 需要部署包和 bit 文件时不要删 |

如果 `folding_config.json` 不存在，某些 FINN 版本可以自动估算；若配置报找不到文件，先去掉该项或让 FINN 自动生成。不要复制别人的 folding 数值到不同网络，因为每层的 `PE`、`SIMD`、FIFO 深度和节点数量都不同。

### 3.6 构建、估算与折叠

在 FINN 根目录执行：

```bash
./run-docker.sh build_dataflow /home/alex/Documents/program/MINIST/fpga/finn_mnist  # 用 FINN Docker 构建项目数据流
```

第一次建议只生成估算报告，先看：

- 每层是否能被推断为 FINN 硬件层。
- 最大的一层是哪个卷积或 `MVAU`。
- BRAM、DSP、LUT 估算是否超过最终器件容量。
- 设计的时钟周期是否可达。
- `PE` 和 `SIMD` 是否导致过度并行。

调节原则：

1. 优先降低大层的 `PE`（并行输出通道数）。
2. 再降低 `SIMD`（并行输入通道数）。
3. 保持 FIFO 足够深，避免数据流断流。
4. 对全连接层单独调节，因为 `3136→128` 是参数最多的层。
5. 先让 `stitched_ip_rtlsim` 通过，再生成 bitstream。
6. 不把“估算通过”当作“时序收敛”的证明；必须看 Vivado 实现报告。

FINN 官方的端到端 CNN notebook 还演示了 `MinimizeWeightBitWidth`、`PE`/`SIMD` 设置、FIFO 深度和逐层折叠。对自定义网络，最可靠的做法是从官方 CNN notebook 的工作流开始，逐段替换成 `DigitCNN`，而不是只照抄一个 JSON 就运行。

### 3.7 FINN 板端接入

FINN 构建成功后通常会生成 bitstream、硬件描述、驱动和部署包。官方示例提供的 PYNQ 驱动只是便捷适配层；生成的网络硬件本质上通过 AXI Stream 连接，可集成到足够资源的自定义 FPGA（以对应 FINN 版本和 IP 接口说明为准）。部署时保持以下数据契约：

```text
当前验证模型输入：uint8 或量化后的 28×28 单通道图像
当前验证模型输出：10 个量化类别分数
最终解密模型：输入加密样本窗口，输出恢复信号或密钥张量
```

FINN 生成的驱动会处理打包后的 AXI Stream 数据。除非确有需要，不要手工按位拆分输入。手工解码极易把 `0..255`、`0..1`、量化缩放和字节序混在一起。若为自定义平台编写 DMA/驱动，至少验证：

- 向 FPGA 发送一段已知加密序列后，输出哈希/BER 与 Python 模拟一致。
- 连续发送多个窗口时，输入输出 FIFO 和序列号不会错位。
- 批量尺寸或最后一窗不满时不会读错或重复输出。
- 通过 AXI Stream 的输入布局与 FINN-ONNX 图输入布局一致。

### 3.8 FINN 调试清单

- **转换器说模型不是 FINN-ONNX**：回到 `export_qonnx`，检查 `ConvertQONNXtoFINN`，不要直接把 PyTorch `.pth` 当 ONNX。
- **找不到量化器模块**：在 FINN Docker 内执行，确认 Brevitas/QONNX 与 FINN 兼容。
- **Flatten/Linear 推断失败**：参考官方 CNN notebook 的 reshape 与全连接整理方式。
- **4W4A 准确率不足**：先试 8W8A，再逐步降位；检查输入范围和训练集划分。
- **资源估算超限**：降低 PE/SIMD，增加 FIFO，减少并行层，必要时降低时钟目标。
- **RTL 仿真通过但板上失败**：先确认驱动和部署包没有使用另一份旧 bit 文件。
- **吞吐看似很低**：区分单次推理延迟、连续吞吐、主机数据准备和 Python 调度开销。

---

## 4. 路线三：Vitis AI Quantizer + DPU + VART

### 4.1 原理与适用性

Vitis AI 的主流板端推理结构是：

1. PyTorch 模型通过 `pytorch_nndct` 做浮点分析和 INT8 量化。
2. 校准集前向传播，记录各层激活和权重分布。
3. 量化器生成 `.xmodel`。
4. `vai_c_xir` 针对具体 DPU 架构编译 `.xmodel`。
5. 目标平台镜像中的 DPU 运行时通过 VART 加载并执行模型。

本项目的卷积、ReLU、池化和全连接都可以尝试映射到 DPU。Vitis AI 的流程成熟、接口统一，尤其适合未来扩展为摄像头、视频流或多模型应用。

但 DPU 只能执行编译器支持的神经网络图。混沌系统本身的迭代方程、帧同步、纠错和解扰运算不应强行伪装成 CNN 层；更合理的划分是“CPU/PL 负责协议和状态，DPU 负责神经网络密钥模块”，并测量两部分之间的数据往返。

但对小模型要注意两个事实：

- DPU 的主要优势在处理大量 CNN 计算；单张 MNIST 的输入只有 784 字节，启动、DMA 和 Python 调度可能比实际计算更耗时。
- DPU 的编译目标是固定架构。`arch.json` 必须和最终平台的 DPU 配置匹配，不能借用其他平台的架构文件。

### 4.2 版本策略

本文以 **Vitis AI 5.0** 为参考。官方 release notes 显示其验证环境与 Vivado/Vitis/PetaLinux **2025.1** 对应，板端内核为 6.12 一类配置。Vitis AI 5.0 还明确说明该版本未启用 `vitis_ai_library`，因此本教程只使用 PyTorch Quantizer、编译器、`.xmodel` 和 VART，不依赖旧版 AI Library。

不要混用以下组件：

- 主机 Docker 中的 Vitis AI 版本。
- 生成 `.xmodel` 的 Quantizer 版本。
- `vai_c_xir` 编译器版本。
- 目标平台的 PetaLinux 镜像和 DPU 运行时版本。
- 目标平台上的 DPU `arch.json` 对应的位流和内核。

最佳做法是直接使用同一条官方 release 中的 Docker 镜像、目标平台镜像和 DPU 架构文件。

Vitis AI 支持多种 DPU 架构，不同架构的指令集、位流和运行时并不通用。目标架构必须从最终平台的 `arch.json`、平台说明或厂商支持包中取得，不能只根据“都使用 Vitis AI”就互换模型。

### 4.3 准备 Docker 与模型

官方 PyTorch 量化示例使用 `xilinx/vitis-ai-pytorch-cpu:latest`。为了可复现，生产环境应把 `latest` 改成明确的 Vitis AI 版本标签。示例启动方式：

```bash
docker pull xilinx/vitis-ai-pytorch-cpu:latest  # 拉取官方 PyTorch 量化镜像
docker run --rm -it -v /home/alex/Documents/program/MINIST:/workspace -w /workspace xilinx/vitis-ai-pytorch-cpu:latest  # 挂载项目并进入容器
```

在容器中先运行浮点模型检查。DPU 编译器对算子顺序敏感，卷积、BN、ReLU 的融合往往比打散节点更容易量化。当前模型没有 BatchNorm，只有 Conv、ReLU、MaxPool、Linear，结构相对简单。

建议先用官方 `Inspector` 检查目标 DPU 是否接受当前拓扑：

```python
import torch  # 导入 PyTorch 用于构造示例输入
from pytorch_nndct.apis import Inspector  # 导入 Vitis AI 模型检查器

from deploy_model import load_deploy_model  # 导入无 Dropout 部署模型


model = load_deploy_model("outputs/best_model.pth", "cpu")  # 加载固定部署模型
target = "<TARGET_FROM_ARCH_JSON>"  # 指定与最终平台完全一致的 DPU 目标名
inspector = Inspector(target)  # 创建目标架构检查器
inspector.inspect(model, (torch.randn(1, 1, 28, 28),), device="cpu")  # 检查层是否可映射
```

这段代码假定当前目录包含 `deploy_model.py`；也可以在容器中把 `fpga` 目录加入 `PYTHONPATH`。完整量化脚本使用 `python fpga/vitis_quantize.py` 启动时，Python 会自动把脚本所在的 `fpga` 目录加入搜索路径。

真实目标名应从最终平台随附的 `arch.json`、平台支持包或 Vitis AI 编译器 arch 目录确认；不预设任何 FPGA 型号。

### 4.4 量化、校准、测试与导出

下面脚本可保存为 `fpga/vitis_quantize.py`。它分三个阶段，和官方 `float`、`calib`、`test` 模式一致：

- `calib`：只做前向校准和导出量化配置，不计算最终准确率。
- `test`：评估量化模型准确率，达标后可导出 `.xmodel`。
- `float`：先用浮点模型检查量化前后的基线。

```python
import argparse  # 导入命令行参数工具
from pathlib import Path  # 导入路径工具

import torch  # 导入 PyTorch
from pytorch_nndct.apis import Inspector  # 导入 Vitis AI Inspector
from pytorch_nndct.apis import torch_quantizer  # 导入 Vitis AI 量化器
from torch import nn  # 导入神经网络模块
from torch.utils.data import DataLoader, Subset  # 导入数据和子集工具
from torchvision import datasets, transforms  # 导入 MNIST 和图像变换

from deploy_model import load_deploy_model  # 导入无 Dropout 部署模型


MEAN = (0.1307,)  # 定义与训练一致的均值
STD = (0.3081,)  # 定义与训练一致的标准差


def parse_args() -> argparse.Namespace:  # 定义命令行参数
    parser = argparse.ArgumentParser(description="Vitis AI DigitCNN INT8 量化")  # 创建解析器
    parser.add_argument("--mode", choices=["float", "calib", "test"], required=True)  # 指定量化阶段
    parser.add_argument("--model", type=Path, default=Path("outputs/best_model.pth"))  # 指定训练模型
    parser.add_argument("--data-dir", type=Path, default=Path("data"))  # 指定 MNIST 数据目录
    parser.add_argument("--config", type=Path, default=None)  # 指定可选量化配置文件
    parser.add_argument("--target", type=str, required=True)  # 指定 DPU 架构目标
    parser.add_argument("--calib-count", type=int, default=500)  # 指定校准图片数量
    parser.add_argument("--test-count", type=int, default=10000)  # 指定量化测试图片数量
    parser.add_argument("--calib-batch-size", type=int, default=32)  # 指定校准阶段的固定批大小
    parser.add_argument("--test-batch-size", type=int, default=1)  # 指定测试阶段的固定批大小以兼容 xmodel 导出
    parser.add_argument("--deploy", action="store_true")  # 指定是否导出 xmodel
    return parser.parse_args()  # 返回解析结果


def make_loader(data_dir: Path, train: bool, count: int, batch_size: int) -> DataLoader:  # 创建固定预处理的数据加载器
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])  # 采用训练时的标准化
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)  # 下载或读取 MNIST
    sample_count = min(count, len(dataset))  # 限制样本数量不超过数据集大小
    generator = torch.Generator().manual_seed(42 if train else 43)  # 为训练校准集和测试集使用不同固定种子
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()  # 选择固定且可复现的随机子集
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=2)  # 返回顺序加载器


@torch.inference_mode()  # 对整个函数关闭梯度计算
def calibration_loop(model: nn.Module, loader: DataLoader) -> None:  # 执行 Vitis AI 校准前向
    model.eval()  # 切换到评估模式
    for images, _ in loader:  # 只使用图像，不需要标签
        model(images)  # 前向一次以收集激活统计量


@torch.inference_mode()  # 对测试函数关闭梯度
def test_quantized(model: nn.Module, loader: DataLoader) -> float:  # 计算量化模型准确率
    model.eval()  # 切换到评估模式
    correct = 0  # 初始化正确数量
    total = 0  # 初始化总数量
    for images, labels in loader:  # 逐批读取测试数据
        output = model(images)  # 执行量化模型推理
        predictions = output.argmax(dim=1)  # 取得最高分类别
        correct += int((predictions == labels).sum())  # 累加正确预测
        total += int(labels.numel())  # 累加样本数
    return correct / total  # 返回准确率


def main() -> None:  # 定义主流程
    args = parse_args()  # 读取参数
    model = load_deploy_model(str(args.model), "cpu")  # 加载无 Dropout 模型
    if args.mode == "float":  # 进入浮点检查分支
        inspector = Inspector(args.target)  # 创建目标架构检查器
        inspector.inspect(model, (torch.randn(1, 1, 28, 28),), device="cpu")  # 打印可映射性检查
        return  # 浮点检查后结束
    fixed_batch = args.calib_batch_size if args.mode == "calib" else args.test_batch_size  # 让占位输入批大小与当前阶段完全一致
    dummy = torch.randn(fixed_batch, 1, 28, 28)  # 创建量化器所需的固定批形状输入
    quantizer = torch_quantizer(  # 创建 Vitis AI 量化器
        args.mode,  # 传入 calib 或 test 阶段
        model,  # 传入浮点部署模型
        (dummy,),  # 指定单输入张量
        device=torch.device("cpu"),  # 量化在 CPU 上执行
        quant_config_file=None if args.config is None else str(args.config),  # 传入可选配置文件
        target=args.target,  # 指定 DPU 目标
    )  # 完成量化模型构建
    quant_model = quantizer.quant_model  # 取出可校准/测试的量化模型
    if args.mode == "calib":  # 进入校准阶段
        calibration_loader = make_loader(args.data_dir, train=True, count=args.calib_count, batch_size=args.calib_batch_size)  # 使用训练集图片做固定批大小的无标签校准
        calibration_loop(quant_model, calibration_loader)  # 收集校准统计
        quantizer.export_quant_config()  # 导出后续 test 阶段需要的量化配置
        print("校准完成，量化配置已导出")  # 提示阶段完成
        return  # 校准阶段结束
    test_loader = make_loader(args.data_dir, train=False, count=args.test_count, batch_size=args.test_batch_size)  # 创建与量化图固定批大小一致的测试加载器
    accuracy = test_quantized(quant_model, test_loader)  # 评估量化模型准确率
    print(f"INT8 量化测试准确率：{accuracy:.2%}")  # 输出量化准确率
    if accuracy < 0.98:  # 检查是否达到项目门槛
        raise SystemExit("INT8 准确率低于 98%，请检查校准集、配置和目标 DPU，不要继续部署")  # 阻止不合格模型导出
    if args.deploy:  # 判断是否要求导出模型
        deploy_loader = make_loader(args.data_dir, train=False, count=1, batch_size=1)  # 创建单张输入加载器
        images, _ = next(iter(deploy_loader))  # 取出单张图片
        quant_model(images)  # 按官方流程在 test 模式执行一次前向
        quantizer.export_xmodel()  # 导出 Vitis AI xmodel
        print("xmodel 已导出到量化输出目录")  # 提示导出完成


if __name__ == "__main__":  # 只有直接执行时才启动
    main()  # 调用主流程
```

推荐执行顺序：

```bash
python fpga/vitis_quantize.py --mode float --target <与目标平台一致的DPU目标>  # 先做算子可映射性检查
python fpga/vitis_quantize.py --mode calib --target <与目标平台一致的DPU目标> --calib-count 500  # 用固定校准集收集统计
python fpga/vitis_quantize.py --mode test --target <与目标平台一致的DPU目标> --test-count 10000 --deploy  # 达标后测试并导出
```

不要用一个目标的量化结果去编译另一个目标。量化器和编译器可能都成功，但板端运行会因 DPU 架构不一致而失败或输出错误。

### 4.5 编译 `.xmodel`

量化完成后，先找到实际 `arch.json`：

```bash
find /opt/vitis_ai/compiler/arch -name arch.json -print  # 列出编译器可用的 DPU 架构文件
```

选择与目标平台镜像完全一致的路径，然后执行：

```bash
vai_c_xir -x quantize_result/DigitCNN_int.xmodel -a <目标DPU架构目录>/arch.json -o digitcnn_pt -n digitcnn_pt  # 编译为最终目标 DPU 可执行模型
```

编译输出通常包括 `.xmodel` 和编译报告。必须检查：

- 没有不支持层被静默回退到 CPU，或者明确知道回退发生在哪里。
- 输入输出张量形状是 `1×1×28×28` 和 `1×10`。
- 编译器报告的 DPU 架构与板端镜像一致。
- 编译后模型能在板端运行的 Vitis AI 版本中加载。

编译命令中的输出文件名和目录会随编译器版本略有不同，上传到板卡前应先用 `find digitcnn_pt -name '*.xmodel' -print` 确认实际产物，不要盲目假设只有一种目录布局。

### 4.6 板端运行

将量化后的 `.xmodel` 复制到板卡：

```bash
scp digitcnn_pt/digitcnn_pt.xmodel user@<板卡IP>:/home/user/digitcnn.xmodel  # 上传编译后的模型
scp fpga/vitis_board_predict.py user@<板卡IP>:/home/user/  # 上传板端推理适配脚本
```

板端脚本应尽量只负责：读取模型、创建 VART runner、把输入送入 DPU、取回 10 个输出。下面是一个通用骨架，保存为 `fpga/vitis_board_predict.py`。不同目标平台镜像的 DPU 输入数据类型和 VART 版本可能不同，必须通过 `input_tensor.get_data_type()` 和官方镜像示例确认后再使用。

```python
import json  # 导入 JSON 用于 HTTP 服务
import sys  # 导入系统参数

import numpy as np  # 导入数组库
import vart  # 导入 Vitis AI 板端运行时
import xir  # 导入 XIR 图和子图 API


def find_dpu_subgraph(graph):  # 在 XIR 图中查找 DPU 子图
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()  # 按拓扑顺序获取子图
    for subgraph in subgraphs:  # 遍历所有子图
        if subgraph.has_attr("device"):  # 判断该子图是否绑定 DPU 设备
            return subgraph  # 返回 DPU 子图
    raise RuntimeError("xmodel 中没有找到 DPU 子图")  # 找不到时给出明确错误


def create_runner(xmodel_path: str):  # 创建 DPU runner 和输入输出描述
    graph = xir.Graph.deserialize(xmodel_path)  # 反序列化 xmodel
    subgraph = find_dpu_subgraph(graph)  # 找到可执行 DPU 子图
    runner = vart.Runner.create_runner(subgraph, "run")  # 在默认 DPU 上创建 runner
    input_tensors = runner.get_input_tensors()  # 获取输入张量描述
    output_tensors = runner.get_output_tensors()  # 获取输出张量描述
    return runner, input_tensors, output_tensors  # 返回运行对象和描述


def predict(runner, input_tensors, output_tensors, image: np.ndarray) -> np.ndarray:  # 执行单张 DPU 推理
    input_data = np.ascontiguousarray(image.astype(np.float32))  # 准备 NCHW float32 输入
    output_data = np.empty(tuple(output_tensors[0].dims), dtype=np.float32)  # 分配单输出缓冲区
    job_id = runner.execute_async([input_data], [output_data])  # 异步提交输入输出
    runner.wait(job_id)  # 等待 DPU 完成
    return output_data.copy()  # 返回独立的结果副本


def main() -> None:  # 定义板端入口
    xmodel_path = sys.argv[1] if len(sys.argv) > 1 else "digitcnn.xmodel"  # 从参数读取模型或使用默认路径
    runner, input_tensors, output_tensors = create_runner(xmodel_path)  # 创建 DPU runner
    image = np.zeros((1, 1, 28, 28), dtype=np.float32)  # 创建一张全黑测试输入
    logits = predict(runner, input_tensors, output_tensors, image)  # 执行测试推理
    print(json.dumps({"logits": logits.reshape(-1).tolist(), "prediction": int(logits.argmax())}))  # 输出 JSON 结果


if __name__ == "__main__":  # 只有直接执行时才运行
    main()  # 启动板端推理
```

VART 的 Python API、输入张量数据类型和 runner 子图筛选在旧镜像上可能有差异。若 `np.float32` 与 `get_data_type()` 不一致，应按最终平台的镜像样例修改，不要强行把 int8 数据声明成 float。优先使用与目标平台匹配的当前 Vitis AI 镜像和样例，而不是归档的旧流程。

### 4.7 Vitis AI 调试清单

- **Inspector 报不支持算子**：先去掉 Dropout 和训练专用层，确认使用 `DeployDigitCNN`；再检查 Flatten/Linear 是否需要改写。
- **compiler 目标不匹配**：板端镜像、DPU `arch.json` 和 `.xmodel` 必须来自同一套硬件设计。
- **量化后分类准确率或 BER 恶化**：校准集应与真实数据分布一致；验证台可用图片，最终项目应使用加密信号样本，并检查是否重复归一化。
- **xmodel 输入范围错误**：确认输入是按 `(uint8/255-0.1307)/0.3081` 还是 `uint8/255`，二者不能互换。
- **导出失败**：校准阶段先导出量化配置，再进入 test 阶段；导出 xmodel 时使用 batch=1。
- **板端加载失败**：先检查镜像版本、DPU 驱动、XRT、VART 和模型编译器版本。
- **单张推理很慢**：用连续多次请求和 warm-up 测量，单独记录主机、DMA、DPU、Python 和 SSH 延迟。

---

## 5. 三种方案的比较与选择方法

### 5.1 按目标选择

| 目标 | 首选 | 原因 |
| --- | --- | --- |
| 用验证模型快速确认定点误差 | hls4ml | Python API 最少，通用 HLS IP 可接入自定义工程 |
| 持续加密样本流的最低延迟 | FINN | 数据流和逐层折叠针对固定网络优化 |
| 视频平台、摄像头和多模块协作 | Vitis AI + DPU | 板端运行时、模型编译和视频平台流程完整 |
| 最大控制、最小面积 | 手写 HLS/RTL | 需要自行完成全部硬件和驱动，本文未选 |
| 含复杂混沌递推或动态协议 | PL/PS 为主 | 三种 NN 编译器都不能自动覆盖任意状态机和数值方程 |

### 5.2 不是性能等同的比较

不要把三条路线的“默认示例”当作公平性能比较。准确度、延迟和资源取决于：

- 权重和激活位宽。
- `PE`、`SIMD`、`ReuseFactor`、`io_type`。
- DPU 队列、DMA 和板卡主频。
- 是否把输入归一化、密钥流生成和视频后处理放在 FPGA。
- 是否统计 Python/SSH/HTTP 开销。
- 是否使用 batch 推理，以及连续样本流是否存在空洞或停顿。

要得到真实结论，必须在自己板卡上用同一段加密样本、同一预处理、同一参考解密结果跑基准，并同时记录 BER、帧延迟和持续吞吐。

---

## 6. 控制预览界面与 FPGA 的集成

### 6.1 保留 PC 端控制界面，板端负责持续推理

推荐结构：

```text
PC 上的 Tkinter app.py（训练、参数、预览）
        │  HTTP/RPC: 加密样本、序列号、帧号和调试 ID
        ▼
板卡上的 Python 服务
        │  选择 hls4ml overlay / FINN 驱动 / VART DPU runner
        ▼
FPGA / DPU
        │  恢复信号或密钥张量
        ▼
服务返回 JSON
        │
        ▼
PC/PS 解扰、重建视频帧、更新预览和误差指标
```

这样做的好处：

- GUI 负责训练参数、板卡选择、视频预览和误码统计，不承担实时硬件的唯一控制面。
- 不必在板卡上运行 Tkinter 图形栈。
- 可把 hls4ml、FINN、Vitis AI 的后端差异封装在一个接口后面。
- 可对同一段加密视频分别测量硬件推理、解扰、编码和显示延迟。

### 6.2 数据接口与返回格式

最终服务应按序列持续传输定长窗口，而不是每帧重新建立一次连接。下面用“当前 28×28 验证模型”的数据包说明字段；实际项目把 `pixels` 换成混沌样本数组，并写明浮点/定点格式：

```json
{
  "sequence_id": 1024,
  "width": 28,
  "height": 28,
  "dtype": "uint8",
  "pixels": [0, 0, 255, 0, 0]
}
```

`pixels` 数组长度应正好为 784，按行优先、从上到下、从左到右排列。真实信号还必须记录采样率、窗口长度、实部/虚部、字节序和同步标记。板端服务明确选择一种内部约定：

- hls4ml/Vitis：`(pixels / 255 - 0.1307) / 0.3081`，然后转换为 NCHW 或 NHWC。
- FINN：`pixels / 255`，然后送入重新训练的量化模型。

返回：

```json
{
  "sequence_id": 1024,
  "output": [0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 0.2, 0.1],
  "output_checksum": 17842,
  "accelerator_ms": 0.42,
  "server_roundtrip_ms": 1.10,
  "backend": "hls4ml"
}
```

验证分类器时可继续把 `output` 解释成 10 个 logits；最终项目应把 `output` 解释成密钥或恢复信号，并用哈希、CRC 和 BER 检查序列是否对应。量化后若密钥流出现一位错误，视频可能出现长时间错位，因此不能只看单帧外观。

### 6.3 GUI 接入策略

现有 `app.py` 的 `_predict_drawing` 目前调用 `predict_tensors`。接入 FPGA 时可以把推理部分抽象成两种后端：

- `LocalPyTorchBackend`：保留当前行为，供训练和离线调试。
- `RemoteFpgaBackend`：把验证画布或真实加密序列发给板卡，并保留 `sequence_id` 以便发现丢包和乱序。

不要在 Tkinter 主线程中直接执行可能阻塞的 SSH/HTTP 请求。应使用现有训练日志读取线程同样的模式，把 FPGA 请求放到后台线程，结果通过队列返回主线程，避免界面卡死。请求超过几毫秒时，按钮应暂时禁用，并显示“FPGA 推理中”。

安全上要把模型权重视为密钥材料而不是可公开常量：bitstream、权重和板端文件通常可以被提取，FPGA 只提供实现载体，不等同于密码学安全边界。若神经网络参与密钥派生，应结合独立密钥存储、唯一 nonce/IV、认证标签和密钥轮换；不要把“权重保密”当作唯一防护。

### 6.4 验证模型画布预处理的一致性

建议提取一个公共函数，保证 PyTorch、hls4ml、FINN、Vitis AI 都使用同一张 28×28 图：

```python
import numpy as np  # 导入数组工具
from PIL import Image  # 导入图像库

from predict import center_on_mnist_canvas  # 复用项目的等比缩放与居中函数


def canvas_to_raw_28(image: Image.Image) -> np.ndarray:  # 把 280×280 灰度画布变成统一的 28×28 uint8
    image = image.convert("L")  # 强制转换为单通道灰度图
    image = center_on_mnist_canvas(image)  # 保持比例缩放并居中到黑底 28×28 画布
    array = np.asarray(image, dtype=np.uint8)  # 转成无符号 8 位数组
    return array[np.newaxis, :, :]  # 增加通道维，得到 1×28×28
```

如果用户通过文件上传白底黑字图片，仍然先按 `predict.py` 的规则做反色；FPGA 服务不应该再猜测颜色。

### 6.5 延迟与视频吞吐应该怎样测

至少报告以下三个时间：

1. **信号窗口延迟**：从第 1 个加密样本进入 FPGA 到密钥/恢复窗口输出的时间。
2. **板端流水线时间**：预处理、模型、解扰、帧缓冲和视频编码的合计时间。
3. **端到端帧延迟**：从采集/上传到预览或显示的时间，包括网络、Python 和 GUI。

对实时视频更重要的是持续吞吐、P99 延迟和抖动。应报告每秒样本数、每秒帧数、帧间隔最大值和连续运行时的 FIFO underflow/overflow；单张 MNIST 的毫秒数只能作为验证索引，不能代表最终视频链路的性能。

---

## 7. 建议的验收流程

### 7.1 软件阶段

1. 运行现有 `python train.py --epochs 5`，确认验证台模型的 test accuracy ≥ 98%。
2. 训练并冻结最终密钥/解密模型，记录模型哈希、输入窗口、输出宽度和取整规则。
3. 生成包含加密输入、参考密钥/恢复信号、序号、CRC 和参考视频帧的共同测试包。
4. 先比较 float32 PyTorch 与定点/INT8 模拟器的 BER、逐位差异和恢复帧 PSNR/SSIM，再进入 RTL 仿真。

### 7.2 硬件阶段

1. 加载 bitstream/overlay，检查设备树、时钟、DMA、帧同步和复位顺序。
2. 用零值、饱和值、随机噪声和已知混沌序列测试边界，防止量化截断掩盖问题。
3. 逐样本比较 FPGA 与模拟器输出，并核对 `sequence_id`、CRC、BER 和是否出现重复/丢包。
4. 恢复一段完整视频并计算 PSNR/SSIM、耗时和任何可见花屏、错帧或颜色偏移。
5. warm-up 后连续运行至少 10–30 分钟，记录样本/秒、帧/秒、平均与 P50/P95/P99 延迟以及 FIFO 异常。
6. 检查温度、降频、内存泄漏、重启恢复和密钥/序列状态重新同步。

### 7.3 验收记录模板

| 项目 | 记录值 |
| --- | --- |
| 目标 FPGA/器件 |  |
| FPGA part |  |
| 工具版本 | Vivado/Vitis/Vitis AI/FINN/hls4ml |
| 模型版本 | checkpoint 文件哈希 |
| 输入格式 | uint8/float32，NCHW/NHWC，范围 |
| 权重量化 | 位宽、对称性、逐张量/逐通道 |
| 激活量化 | 位宽、范围 |
| 资源 | LUT、FF、BRAM、DSP、URAM |
| 时钟 | 目标频率与实际 WNS |
| 验证模型分类准确率（仅代理指标） |  |
| 解密 BER/符号错误率 |  |
| 与 PyTorch 输出逐位/数值差异 |  |
| 恢复视频 PSNR/SSIM |  |
| 丢包、重复、乱序和 CRC 失败数 |  |
| 内核延迟 |  |
| 端到端帧延迟与 P99 抖动 |  |
| 持续样本数/秒、帧数/秒、连续运行时长 |  |

达到以下条件才可认为移植完成：

- 最终解密模型达到预先约定的 BER、PSNR/SSIM 和确定性要求；98% 分类准确率只用于验证台。
- 完整加密视频在多轮连续运行中无丢包、重复、错位或无法恢复的帧。
- GUI 或服务能调整参数、选择板卡、启动流式推理并预览恢复结果。
- 最终信号归一化、取整、字节序和密钥同步规则与训练完全一致。
- 模型、bitstream、DPU 架构和板端运行时版本匹配。
- 有可重复的启动、持续流式推理、密钥重置和故障恢复步骤。

---

## 8. 常见问题总表

| 现象 | 优先排查 | 处理 |
| --- | --- | --- |
| 所有图片都预测同一个数字 | 输入范围、布局、全零/全白 | 用已知 MNIST 图片和原始 logits 对照 |
| PyTorch 99%，FPGA 80% 左右 | 定点位宽、量化校准、输入标准化 | 先恢复 float 仿真，再逐层定量化 |
| PyTorch 与板端 logit 量级差很多 | 输出缩放、反量化、是否再做 Softmax | 先比较 argmax 和排序，再比较数值 |
| hls4ml HLS 仿真很慢 | 固定点仿真、全展开、C 仿真 | 先用小批数据；综合前看估算报告 |
| hls4ml 综合资源爆炸 | `ReuseFactor`、`io_stream`、逐层精度 | 增大复用因子，减少全并行 |
| FINN 无法推断层 | ONNX/QONNX 格式、拓扑 | 回到官方 CNN notebook 对照中间模型 |
| FINN 构建卡在 RTL 仿真 | 波形和 FIFO 调试开启过多 | 先关闭波形保存，保留节点级检查 |
| Vitis AI 编译失败 | arch 目标、输入 batch、算子支持 | 用 Inspector 和官方目标文件重查 |
| 板端 runner 创建失败 | DPU 子图、runtime 版本 | 查看 xmodel 子图并匹配目标平台镜像 |
| DMA/PCIe 传输成为瓶颈 | 单张推理、小批次 | 合并批次、减少 Python 往返 |
| 量化后 BER 增大但分类指标正常 | 密钥位精度、累加截断、输出反量化 | 对密钥/恢复信号单独做逐位回归测试 |
| 视频逐渐花屏或周期性错位 | 序列号、帧边界、FIFO、同步标记 | 长时间抓取输入输出序列并检查空洞 |
| 单帧正确但连续视频失败 | 状态复位、密钥轮换、DMA 环形缓冲 | 划分 PL/PS 状态职责并增加端到端复位 |
| 神经网络输出被读出 | 密钥存储、访问控制、bitstream 保护 | 加入独立密钥管理、nonce、认证与轮换机制 |

---

## 9. 参考资料与版本说明

本文的选择以官方文档和官方示例为主，社区文章只用于理解流程，不作为版本接口的唯一依据。FPGA 工具链对版本非常敏感，复制命令前请先核对你实际安装的版本。

- FINN 官方仓库：<https://github.com/Xilinx/finn>，本文参考 v0.10.1 和主线文档。
- FINN Getting Started：<https://finn.readthedocs.io/en/latest/getting_started.html>。
- FINN 端到端 CNN 示例：<https://github.com/Xilinx/finn/blob/main/notebooks/end2end_example/bnn-pynq/cnv_end2end_example.ipynb>。
- FINN examples：<https://github.com/Xilinx/finn-examples>。
- Brevitas：<https://github.com/Xilinx/brevitas>。
- Brevitas QONNX 导出教程：<https://xilinx.github.io/brevitas/tutorials/onnx_export.html>。
- hls4ml 官方仓库：<https://github.com/fastmachinelearning/hls4ml>，本文编写时正式版本为 v1.3.0。
- hls4ml 文档：<https://fastmachinelearning.org/hls4ml/>。
- hls4ml PyTorch 前端：<https://github.com/fastmachinelearning/hls4ml/blob/main/docs/frontend/pytorch.rst>。
- hls4ml VivadoAccelerator：<https://github.com/fastmachinelearning/hls4ml/blob/main/docs/backend/accelerator.rst>。
- hls4ml tutorials：<https://github.com/fastmachinelearning/hls4ml-tutorial>。
- Vitis AI 官方仓库：<https://github.com/Xilinx/Vitis-AI>，本文编写时正式版本为 v5.0。
- Vitis AI 文档：<https://xilinx.github.io/Vitis-AI/5.0/>。
- Vitis AI MPSoC Quickstart：<https://xilinx.github.io/Vitis-AI/5.0/html/docs/quickstart/mpsoc.html>。
- Vitis AI PyTorch Quantizer 示例：<https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/example/resnet18_quant.py>。
- Vitis AI PyTorch Quantizer 配置文件：<https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/example/pytorch_quantize_config.json>。
- 已归档的 DPU-PYNQ：<https://github.com/Xilinx/DPU-PYNQ>；仅用于理解旧版 PYNQ/DPU 示例，不建议作为 Vitis AI 5.0 的默认路线。

### 9.1 本教程未覆盖的工程细节

- 没有在本文编写环境中实际安装 Vivado/Vitis、生成 bitstream 或上板，因此文中不提供虚构的 LUT、DSP、BRAM、延迟、BER 和视频质量数字。
- 不同 FINN/hls4ml/Vitis AI 版本可能调整函数签名、生成目录和驱动类名；遇到差异时以对应版本的官方示例和生成文件为准。
- 具体板卡的时钟约束、DDR 带宽、DMA、PS 启动、PetaLinux 镜像和许可证需要使用板卡厂商/AMD 官方资料确认。
- 若需要进一步把本项目变成可复现实验，建议在确定最终 FPGA、时钟和 I/O 后补充板卡无关的 target profile，例如 `<target>-<flow>-<quantization>-video-v1`，并把综合报告、BER/PSNR、持续帧率、帧延迟和故障恢复日志一起归档。

