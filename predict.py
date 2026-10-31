import argparse  # 导入命令行参数解析工具，用于接收图片路径和推理选项
import json  # 导入 JSON 工具，用于把推理结果保存为结构化文件
from pathlib import Path  # 导入路径工具，便于创建目录、检查模型和保存结果

import torch  # 导入 PyTorch 主模块，用于加载模型和执行张量计算
from PIL import Image, ImageOps  # 导入 Pillow 图像工具，用于读取、旋转、灰度化和缩放图片
from torchvision import datasets, transforms  # 导入 MNIST 数据集和图像预处理工具

from model import DigitCNN  # 从本地模型文件导入与训练时完全一致的网络结构
from train import choose_device  # 复用训练脚本中的设备自动选择函数


MNIST_MEAN = (0.1307,)  # 定义训练时使用的 MNIST 全局像素均值
MNIST_STD = (0.3081,)  # 定义训练时使用的 MNIST 全局像素标准差


def parse_args():  # 定义推理阶段的命令行参数解析函数
    parser = argparse.ArgumentParser(description="使用训练好的 CNN 模型推理手写数字")  # 创建带程序说明的参数解析器
    parser.add_argument("images", nargs="*", type=Path, help="一张或多张待识别图片的路径")  # 添加可一次传入多张图片的位置参数
    parser.add_argument("--sample-index", type=int, default=None, help="读取 MNIST 测试集的指定样本，编号范围 0 到 9999")  # 添加直接测试数据集样本的可选参数
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST 数据目录")  # 添加 MNIST 数据集的保存目录
    parser.add_argument("--model", type=Path, default=Path("outputs/best_model.pth"), help="训练得到的模型文件路径")  # 添加模型检查点路径
    parser.add_argument("--top-k", type=int, default=3, help="显示概率最高的前几个候选数字")  # 添加候选数量参数
    parser.add_argument("--invert", action="store_true", help="黑白反转，用于识别白底黑字图片")  # 添加适用普通拍照和扫描图片的颜色反转开关
    parser.add_argument("--output-json", type=Path, default=None, help="可选的推理结果 JSON 保存路径")  # 添加结构化结果输出路径
    args = parser.parse_args()  # 解析命令行输入
    if not args.images and args.sample_index is None:  # 检查用户是否提供了任意一种图片来源
        parser.error("请提供图片路径，或使用 --sample-index 指定 MNIST 样本")  # 在没有输入图片时给出明确提示
    if args.sample_index is not None and not 0 <= args.sample_index < 10000:  # 检查 MNIST 测试集样本编号是否有效
        parser.error("--sample-index 必须在 0 到 9999 之间")  # 拒绝超出测试集范围的编号
    if not 1 <= args.top_k <= 10:  # 检查候选类别数量是否在合理范围内
        parser.error("--top-k 必须在 1 到 10 之间")  # 拒绝无法从 10 个数字类别中取出的数量
    return args  # 返回通过基础校验的参数


def build_transform():  # 定义创建推理图像预处理流程的函数
    return transforms.Compose(  # 组合与训练阶段一致的预处理步骤
        [  # 开始定义预处理步骤列表
            transforms.ToTensor(),  # 把 0 到 255 的灰度图转换为 0 到 1 的单通道张量
            transforms.Normalize(MNIST_MEAN, MNIST_STD),  # 使用训练集均值和标准差标准化每个像素
        ]  # 结束预处理步骤列表
    )  # 返回完整的预处理流程


def load_model(model_path: Path, device: torch.device):  # 定义加载模型结构和参数检查点的函数
    if not model_path.exists():  # 检查模型文件是否存在
        raise FileNotFoundError(f"找不到模型文件：{model_path}")  # 文件不存在时给出可直接定位问题的错误
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)  # 安全加载模型权重和记录的训练信息
    if "model_state_dict" not in checkpoint:  # 检查检查点是否符合本项目保存格式
        raise ValueError("模型文件缺少 model_state_dict，可能不是本项目生成的检查点")  # 对格式错误的模型给出明确提示
    model = DigitCNN().to(device)  # 创建与训练时相同的网络并移动到计算设备
    model.load_state_dict(checkpoint["model_state_dict"])  # 把训练好的参数加载到网络
    model.eval()  # 切换到评估模式，关闭 Dropout 并固定批量归一化等行为
    return model, checkpoint  # 返回准备好推理的模型和附带训练信息的检查点


def center_on_mnist_canvas(image: Image.Image) -> Image.Image:  # 定义保持比例缩放并把数字居中放到 28×28 画布的函数
    image = image.copy()  # 复制图片以避免原地修改调用方持有的对象
    image.thumbnail((28, 28), Image.Resampling.LANCZOS)  # 等比例缩小图片，使较长边不超过 MNIST 的 28 像素
    canvas = Image.new("L", (28, 28), color=0)  # 创建黑底单通道画布，像素值 0 表示纯黑
    left = (28 - image.width) // 2  # 计算图片在水平方向居中时的左边界
    top = (28 - image.height) // 2  # 计算图片在垂直方向居中时的上边界
    canvas.paste(image, (left, top))  # 把缩放后的图片粘贴到画布中央
    return canvas  # 返回尺寸和布局符合 MNIST 风格的图片


def preprocess_pil_image(image: Image.Image, transform, invert: bool) -> torch.Tensor:  # 定义把 PIL 图片转换为模型输入张量的函数
    image = ImageOps.exif_transpose(image)  # 根据照片 EXIF 信息纠正手机图片可能存在的旋转方向
    image = ImageOps.grayscale(image)  # 把 RGB、RGBA 或其他颜色模式统一转换为单通道灰度图
    if invert:  # 判断是否需要处理白底黑字图片
        image = ImageOps.invert(image)  # 反转亮度，使普通白纸黑字变成 MNIST 风格的黑底白字
    image = center_on_mnist_canvas(image)  # 等比例缩放并居中添加黑色边距，避免直接拉伸造成字形变形
    return transform(image)  # 转换为标准化后的 1×28×28 浮点张量


def preprocess_file(image_path: Path, transform, invert: bool) -> torch.Tensor:  # 定义从磁盘图片路径生成模型输入张量的函数
    with Image.open(image_path) as image:  # 打开图片文件，并确保使用结束后及时关闭文件句柄
        return preprocess_pil_image(image, transform, invert)  # 完成方向修正、灰度化、缩放、居中和标准化


@torch.inference_mode()  # 推理模式比普通 no_grad 更严格，可禁止构建自动求导图并降低内存占用
def predict_tensors(model, tensors, device: torch.device, top_k: int):  # 定义对一组预处理图片执行批量推理的函数
    batch = torch.stack(tensors).to(device)  # 把多张 1×28×28 图片堆叠成批量×1×28×28 张量并移动到设备
    logits = model(batch)  # 通过前向传播得到每张图片的 10 个类别分数
    probabilities = torch.softmax(logits, dim=1)  # 使用 Softmax 把类别分数转换为总和为 1 的概率
    top_probabilities, top_classes = probabilities.topk(top_k, dim=1)  # 选出每张图片概率最高的前 K 个候选类别
    top_probabilities = top_probabilities.cpu().tolist()  # 把 GPU 上的概率复制到 CPU 并转换为 Python 列表
    top_classes = top_classes.cpu().tolist()  # 把 GPU 上的类别编号复制到 CPU 并转换为 Python 列表
    results = []  # 初始化最终结果列表
    for classes, scores in zip(top_classes, top_probabilities):  # 逐张整理类别编号和对应概率
        candidates = [{"digit": digit, "probability": probability} for digit, probability in zip(classes, scores)]  # 创建有序的前 K 个候选结果
        results.append({"prediction": classes[0], "confidence": scores[0], "top_k": candidates})  # 记录最终预测、置信度和完整候选列表
    return results  # 返回每张图片对应的结构化推理结果


def build_sources(args):  # 定义收集普通图片和 MNIST 测试样本的函数
    transform = build_transform()  # 创建一次预处理流程，并让所有输入图片共用
    tensors = []  # 初始化模型输入张量列表
    descriptions = []  # 初始化与张量一一对应的来源说明列表
    for image_path in args.images:  # 依次处理命令行传入的每张图片
        tensors.append(preprocess_file(image_path, transform, args.invert))  # 把普通图片转换为模型输入张量
        descriptions.append({"source": str(image_path), "ground_truth": None})  # 记录图片来源，普通图片没有已知标签
    if args.sample_index is not None:  # 判断是否还需要读取一张 MNIST 测试样本
        test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=None)  # 读取不经过变换的原始测试集图片
        sample_image, ground_truth = test_dataset[args.sample_index]  # 按编号取出图片和真实标签
        tensors.append(preprocess_pil_image(sample_image, transform, invert=False))  # 把原始测试样本转换为模型输入
        descriptions.append({"source": f"MNIST test[{args.sample_index}]", "ground_truth": ground_truth})  # 记录样本来源和用于核验的真实标签
    return tensors, descriptions  # 返回模型输入与对应来源信息


def save_results(path: Path, checkpoint, descriptions, predictions) -> None:  # 定义把推理结果保存为 JSON 文件的函数
    path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建结果文件所需的父目录
    records = []  # 初始化用于序列化输出的记录列表
    for description, prediction in zip(descriptions, predictions):  # 把每张图片的来源信息和推理结果合并
        records.append({**description, **prediction})  # 用字典展开运算合并相同层级的字段
    content = {"validation_accuracy": checkpoint["validation_accuracy"], "predictions": records}  # 组合模型信息与全部推理记录
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")  # 以 UTF-8 和易读缩进写出 JSON 文件


def main() -> None:  # 定义推理脚本的整体入口流程
    args = parse_args()  # 读取并校验命令行参数
    device = choose_device()  # 自动选择 CUDA、MPS 或 CPU
    model, checkpoint = load_model(args.model, device)  # 加载最佳模型参数并切换到评估模式
    tensors, descriptions = build_sources(args)  # 把全部输入图片转换为可批量推理的张量
    predictions = predict_tensors(model, tensors, device, args.top_k)  # 一次性完成整批图片推理
    print(f"使用设备: {device}")  # 显示本次推理使用的计算设备
    print(f"模型验证准确率: {checkpoint['validation_accuracy']:.2%}")  # 显示保存模型时在验证集上的准确率
    for description, prediction in zip(descriptions, predictions):  # 逐张打印容易阅读的预测结果
        candidates = ", ".join(f"{item['digit']}: {item['probability']:.2%}" for item in prediction["top_k"])  # 组合前 K 个候选数字和概率文本
        truth_text = ""  # 默认不显示真实标签，因为普通图片没有标签
        if description["ground_truth"] is not None:  # 判断当前结果是否来自带标签的 MNIST 测试样本
            truth_text = f" | 真实标签 {description['ground_truth']}"  # 为测试样本补充真实标签
        print(f"{description['source']}: 预测为 {prediction['prediction']} | 置信度 {prediction['confidence']:.2%} | 前 {args.top_k} 名 {candidates}{truth_text}")  # 输出最终预测、置信度、候选和可选真实标签
    if args.output_json is not None:  # 判断用户是否要求保存结构化结果
        save_results(args.output_json, checkpoint, descriptions, predictions)  # 把全部推理结果保存为 JSON 文件
        print(f"推理结果已保存到 {args.output_json}")  # 提示结果文件的保存位置


if __name__ == "__main__":  # 只有直接运行本文件时才启动推理流程
    main()  # 调用主函数执行推理
