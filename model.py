import torch  # 导入 PyTorch 主模块，用于张量运算
from torch import nn  # 导入神经网络模块，并简写为 nn


class DigitCNN(nn.Module):  # 定义用于识别 0 到 9 的卷积神经网络
    def __init__(self) -> None:  # 初始化网络的各个层
        super().__init__()  # 初始化父类，使网络具备 PyTorch 模块能力
        self.features = nn.Sequential(  # 把负责提取图像特征的层按顺序组合
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 用 32 个 3×3 卷积核提取笔画边缘等基础特征，并保持 28×28 尺寸
            nn.ReLU(inplace=True),  # 用 ReLU 引入非线性，原地计算以节省少量内存
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 把 32 个特征图进一步组合成更丰富的局部特征
            nn.ReLU(inplace=True),  # 再次用 ReLU 保留正响应并抑制负响应
            nn.MaxPool2d(kernel_size=2),  # 用 2×2 最大池化把特征图缩小到 14×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 用 64 个卷积核学习更高级的数字部件
            nn.ReLU(inplace=True),  # 对新增的 64 个特征图应用 ReLU
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 进一步组合高级特征以提高分类能力
            nn.ReLU(inplace=True),  # 对最后一组卷积结果应用 ReLU
            nn.MaxPool2d(kernel_size=2),  # 再次池化，把特征图缩小到 7×7
            nn.Dropout2d(0.25),  # 训练时随机屏蔽 25% 的特征图，降低过拟合风险
        )  # 结束特征提取部分的定义
        self.classifier = nn.Sequential(  # 把提取出的特征映射到 10 个数字类别
            nn.Flatten(),  # 把 64×7×7 的三维特征展平成一维向量
            nn.Linear(64 * 7 * 7, 128),  # 用 3136 到 128 的全连接层学习全局组合特征
            nn.ReLU(inplace=True),  # 给全连接层加入非线性表达能力
            nn.Dropout(0.5),  # 训练时随机屏蔽 50% 的神经元，进一步抑制过拟合
            nn.Linear(128, 10),  # 输出 10 个分数，分别对应数字 0 到 9
        )  # 结束分类器部分的定义

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义输入图像到输出类别分数的前向计算
        x = self.features(x)  # 从形状为 (批量, 1, 28, 28) 的图像中提取特征
        x = self.classifier(x)  # 把展平后的特征转换为形状为 (批量, 10) 的类别分数
        return x  # 返回未经过 Softmax 的原始类别分数 logits
