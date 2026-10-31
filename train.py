import argparse  # 导入命令行参数解析工具，便于修改训练配置
import json  # 导入 JSON 工具，用于保存训练历史和最终指标
import random  # 导入 Python 随机数模块，用于设置可复现的随机种子
import time  # 导入时间模块，用于统计每个 epoch 的耗时
from pathlib import Path  # 导入路径工具，以跨平台方式创建目录和文件

import torch  # 导入 PyTorch 主模块，用于张量计算和设备管理
from torch import nn  # 导入神经网络模块，用于构建损失函数
from torch.optim import Adam  # 导入 Adam 优化器，用于更新网络参数
from torch.utils.data import DataLoader, random_split  # 导入数据加载器和随机划分工具
from torchvision import datasets, transforms  # 导入 MNIST 数据集和图像预处理工具

from model import DigitCNN  # 从本地模型文件导入数字识别网络


MNIST_MEAN = (0.1307,)  # 定义 MNIST 像素的全局均值，标准化时使用
MNIST_STD = (0.3081,)  # 定义 MNIST 像素的全局标准差，标准化时使用


def set_seed(seed: int) -> None:  # 定义统一设置随机种子的函数
    random.seed(seed)  # 固定 Python 自带随机数生成器的种子
    torch.manual_seed(seed)  # 固定 PyTorch CPU 随机数生成器的种子
    torch.cuda.manual_seed_all(seed)  # 固定所有 GPU 随机数生成器的种子
    torch.backends.cudnn.deterministic = True  # 让 cuDNN 尽量选择结果可复现的算法
    torch.backends.cudnn.benchmark = False  # 关闭自动寻找最快算法的行为，以增强可复现性


def choose_device() -> torch.device:  # 定义自动选择计算设备的函数
    if torch.cuda.is_available():  # 判断当前环境是否有可用的 NVIDIA GPU
        return torch.device("cuda")  # 有 GPU 时优先使用 CUDA 加速
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # 判断是否支持 Apple Silicon 的 MPS 加速
    if mps_available:  # 判断是否检测到可用的 Apple GPU
        return torch.device("mps")  # 有 Apple GPU 时使用 MPS
    return torch.device("cpu")  # 没有可用加速器时使用 CPU


def build_data_loaders(data_dir: Path, batch_size: int, workers: int, device: torch.device, seed: int):  # 定义创建训练、验证和测试加载器的函数
    transform = transforms.Compose(  # 组合多个图像预处理步骤
        [  # 开始定义预处理步骤列表
            transforms.ToTensor(),  # 把 0 到 255 的 PIL 图像转换为 0 到 1 的 PyTorch 张量
            transforms.Normalize(MNIST_MEAN, MNIST_STD),  # 用 MNIST 统计量标准化像素，使训练更稳定
        ]  # 结束预处理步骤列表
    )  # 完成图像预处理流程
    train_dataset = datasets.MNIST(  # 创建 MNIST 训练集对象
        root=data_dir,  # 指定数据集下载和保存的目录
        train=True,  # 指定读取包含 60000 张图片的训练集
        download=True,  # 首次运行时用 Python 自动下载并校验数据
        transform=transform,  # 对每张训练图片执行上述预处理
    )  # 完成训练集创建
    test_dataset = datasets.MNIST(  # 创建 MNIST 测试集对象
        root=data_dir,  # 继续使用相同的数据保存目录
        train=False,  # 指定读取包含 10000 张图片的测试集
        download=True,  # 如果训练集下载时尚未覆盖测试文件，这里会自动补齐
        transform=transform,  # 对每张测试图片使用完全相同的预处理
    )  # 完成测试集创建
    train_size = int(len(train_dataset) * 0.9)  # 用 90% 的原始训练数据更新模型参数
    validation_size = len(train_dataset) - train_size  # 把剩余 10% 的原始训练数据作为验证集
    split_generator = torch.Generator().manual_seed(seed)  # 用固定种子保证每次得到相同的训练和验证划分
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size], generator=split_generator)  # 随机划分训练集与验证集，二者仍共享相同的图像预处理
    pin_memory = device.type == "cuda"  # 仅在 CUDA 场景开启锁页内存以加快主机到 GPU 的传输
    train_loader = DataLoader(  # 创建训练数据批次加载器
        train_dataset,  # 指定训练集
        batch_size=batch_size,  # 每个批次包含 64 张图片
        shuffle=True,  # 每个 epoch 打乱训练样本顺序，减少固定顺序带来的偏差
        num_workers=workers,  # 指定后台读取数据的进程数
        pin_memory=pin_memory,  # 根据设备决定是否使用锁页内存
        persistent_workers=workers > 0,  # 使用多进程时让读取进程在 epoch 之间保留下来
    )  # 完成训练加载器创建
    validation_loader = DataLoader(  # 创建验证数据批次加载器
        validation_dataset,  # 指定从训练集划分出来的验证集
        batch_size=batch_size,  # 验证时使用与训练相同的批次大小
        shuffle=False,  # 验证不需要打乱顺序，固定顺序便于比较各轮结果
        num_workers=workers,  # 使用相同的后台读取进程数
        pin_memory=pin_memory,  # 根据设备决定是否使用锁页内存
        persistent_workers=workers > 0,  # 使用多进程时保留验证数据读取进程
    )  # 完成验证加载器创建
    test_loader = DataLoader(  # 创建测试数据批次加载器
        test_dataset,  # 指定测试集
        batch_size=batch_size,  # 测试时使用与训练相同的批次大小
        shuffle=False,  # 测试不需要打乱顺序，固定顺序便于复现结果
        num_workers=workers,  # 使用相同的后台读取进程数
        pin_memory=pin_memory,  # 根据设备决定是否使用锁页内存
        persistent_workers=workers > 0,  # 使用多进程时保留测试数据读取进程
    )  # 完成测试加载器创建
    return train_loader, validation_loader, test_loader  # 返回训练、验证和测试加载器


def train_one_epoch(model, loader, optimizer, criterion, device):  # 定义完成一个完整训练 epoch 的函数
    model.train()  # 把网络切换到训练模式，使 Dropout 正常工作
    total_loss = 0.0  # 初始化当前 epoch 的累计损失
    correct = 0  # 初始化预测正确的样本数量
    total = 0  # 初始化已经处理的样本数量
    for images, labels in loader:  # 逐批次读取图像和真实标签
        images = images.to(device, non_blocking=True)  # 把图像移动到 GPU 或 CPU 计算设备
        labels = labels.to(device, non_blocking=True)  # 把标签移动到同一计算设备
        optimizer.zero_grad(set_to_none=True)  # 清空上一批次留下的梯度，并把已用梯度内存置空
        logits = model(images)  # 通过前向传播得到每张图片的 10 个类别分数
        loss = criterion(logits, labels)  # 计算预测分数与真实标签之间的交叉熵损失
        loss.backward()  # 通过反向传播计算每个网络参数的梯度
        optimizer.step()  # 根据梯度更新网络参数
        batch_size = labels.size(0)  # 取出当前批次的实际样本数
        total_loss += loss.item() * batch_size  # 累加批次平均损失乘以样本数，便于最后求总体平均
        predictions = logits.argmax(dim=1)  # 选择分数最高的类别作为预测数字
        correct += (predictions == labels).sum().item()  # 统计当前批次预测正确的样本数
        total += batch_size  # 累加当前批次处理过的样本数
    return total_loss / total, correct / total  # 返回本轮平均训练损失和训练准确率


@torch.no_grad()  # 评估阶段关闭自动求导，减少内存占用并提升速度
def evaluate(model, loader, criterion, device):  # 定义在验证集或测试集上评估网络的函数
    model.eval()  # 把网络切换到评估模式，关闭 Dropout
    total_loss = 0.0  # 初始化累计评估损失
    correct = 0  # 初始化预测正确的样本数量
    total = 0  # 初始化已经处理的样本数量
    for images, labels in loader:  # 逐批次读取数据
        images = images.to(device, non_blocking=True)  # 把图像移动到指定计算设备
        labels = labels.to(device, non_blocking=True)  # 把标签移动到指定计算设备
        logits = model(images)  # 只进行前向传播，得到类别分数
        loss = criterion(logits, labels)  # 计算当前批次的交叉熵损失
        batch_size = labels.size(0)  # 取出当前批次的实际样本数
        total_loss += loss.item() * batch_size  # 按样本数累加损失
        predictions = logits.argmax(dim=1)  # 把分数最高的类别作为预测结果
        correct += (predictions == labels).sum().item()  # 累加预测正确的样本数
        total += batch_size  # 累加已处理的样本数
    return total_loss / total, correct / total  # 返回平均损失和准确率


def save_json(path: Path, content) -> None:  # 定义把 Python 内容保存为便于阅读的 JSON 文件的函数
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")  # 以 UTF-8 编码写出带缩进的 JSON 文本


def parse_args():  # 定义读取命令行配置的函数
    parser = argparse.ArgumentParser(description="从零训练 MNIST 手写数字识别 CNN")  # 创建带程序说明的参数解析器
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST 数据保存目录")  # 添加数据集保存路径参数
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="模型和结果保存目录")  # 添加训练产物保存路径参数
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数，默认 5 轮足以超过 98%%")  # 添加训练轮数参数，百分号需写成 %% 以兼容 argparse 格式
    parser.add_argument("--batch-size", type=int, default=64, help="每个批次的样本数")  # 添加批次大小参数
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam 的学习率")  # 添加学习率参数
    parser.add_argument("--workers", type=int, default=2, help="读取数据的后台进程数")  # 添加数据加载进程数参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于复现实验")  # 添加随机种子参数
    parser.add_argument("--target-accuracy", type=float, default=0.98, help="项目要求达到的测试准确率")  # 添加目标准确率参数
    args = parser.parse_args()  # 解析用户输入的命令行参数
    if args.epochs < 1:  # 检查训练轮数是否为正数
        parser.error("--epochs 必须大于等于 1")  # 拒绝无法正常完成训练的轮数
    if args.batch_size < 1:  # 检查批次大小是否为正数
        parser.error("--batch-size 必须大于等于 1")  # 拒绝无法形成有效批次的参数
    return args  # 返回通过基础检查的参数


def main() -> int:  # 定义程序主流程，并返回进程退出码
    args = parse_args()  # 读取命令行参数
    set_seed(args.seed)  # 使用指定种子保证实验尽量可复现
    device = choose_device()  # 自动选择 CUDA、MPS 或 CPU
    args.output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录，并允许目录已经存在
    print(f"使用设备: {device}")  # 显示本次训练使用的设备
    train_loader, validation_loader, test_loader = build_data_loaders(args.data_dir, args.batch_size, args.workers, device, args.seed)  # 创建自动下载数据后的三个加载器
    model = DigitCNN().to(device)  # 创建网络并把全部参数放到选定设备
    criterion = nn.CrossEntropyLoss()  # 创建交叉熵损失，它内部会先对 logits 做 Softmax
    optimizer = Adam(model.parameters(), lr=args.learning_rate)  # 创建 Adam 优化器并绑定全部可训练参数
    best_accuracy = 0.0  # 记录训练过程中最高的验证集准确率
    history = []  # 保存每个 epoch 的损失和准确率
    best_model_path = args.output_dir / "best_model.pth"  # 定义最佳模型检查点的保存路径
    start_time = time.perf_counter()  # 记录整体训练开始的时间
    for epoch in range(1, args.epochs + 1):  # 按指定轮数依次训练
        epoch_start = time.perf_counter()  # 记录当前 epoch 开始的时间
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)  # 完成一轮参数更新
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)  # 在验证集上观察泛化效果并决定是否保存模型
        elapsed = time.perf_counter() - epoch_start  # 计算当前 epoch 实际耗时
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | 训练损失 {train_loss:.4f} | 训练准确率 {train_accuracy:.2%} | 验证损失 {validation_loss:.4f} | 验证准确率 {validation_accuracy:.2%} | 用时 {elapsed:.1f}s")  # 输出本轮完整指标
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy, "seconds": elapsed})  # 记录本轮指标
        save_json(args.output_dir / "history.json", history)  # 每轮覆盖写入训练历史，避免中途异常时丢失全部记录
        if validation_accuracy > best_accuracy:  # 判断当前模型是否刷新了最佳验证准确率
            best_accuracy = validation_accuracy  # 更新最佳验证准确率
            checkpoint = {"model_state_dict": model.state_dict(), "validation_accuracy": best_accuracy, "epoch": epoch, "normalization_mean": MNIST_MEAN, "normalization_std": MNIST_STD}  # 打包模型参数和预测时需要的元数据
            torch.save(checkpoint, best_model_path)  # 保存当前表现最好的模型
            print(f"已保存新的最佳模型，准确率 {best_accuracy:.2%}")  # 提示最佳模型已更新
    total_seconds = time.perf_counter() - start_time  # 计算全部训练所需时间
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)  # 重新读取最佳模型检查点
    model.load_state_dict(checkpoint["model_state_dict"])  # 把最佳参数加载回网络
    _, final_accuracy = evaluate(model, test_loader, criterion, device)  # 独立复核最佳模型的最终测试准确率
    metrics = {"test_accuracy": final_accuracy, "target_accuracy": args.target_accuracy, "target_reached": final_accuracy >= args.target_accuracy, "best_epoch": checkpoint["epoch"], "training_seconds": total_seconds, "device": str(device), "model_path": str(best_model_path)}  # 汇总最终结果
    save_json(args.output_dir / "metrics.json", metrics)  # 把最终结果写入 JSON 文件
    print(f"最终测试准确率: {final_accuracy:.2%}")  # 显示经过最佳模型复核的准确率
    if final_accuracy >= args.target_accuracy:  # 判断是否达到项目要求的 98%
        print(f"目标已达成：准确率不低于 {args.target_accuracy:.0%}")  # 输出成功信息
        return 0  # 返回 0 表示程序成功
    print(f"目标未达成：请增加 --epochs，或检查数据是否完整")  # 输出失败原因和调整建议
    return 1  # 返回非零退出码，便于自动化脚本发现失败


if __name__ == "__main__":  # 只有直接运行本文件时才启动训练
    raise SystemExit(main())  # 用主函数返回值作为程序退出码
