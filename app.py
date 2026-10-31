import json  # 导入 JSON 工具，用于读取训练完成后的最终指标
import queue  # 导入线程安全队列，用于把后台训练日志传给界面线程
import re  # 导入正则表达式工具，用于解析训练进度和准确率
import subprocess  # 导入子进程工具，用于在后台启动训练脚本
import sys  # 导入 Python 系统模块，用于获取当前解释器路径
import threading  # 导入线程工具，用于持续读取训练子进程输出
from pathlib import Path  # 导入路径工具，用于定位项目文件和模型

import tkinter as tk  # 导入 Tkinter 主模块，用于创建桌面窗口
from tkinter import messagebox, ttk  # 导入消息框和主题控件，用于提示信息与美化界面
from PIL import Image, ImageDraw  # 导入 Pillow 绘图工具，用于保存画布中的手写图像

from predict import build_transform, load_model, predict_tensors, preprocess_pil_image  # 复用推理脚本中的数据预处理和模型推理函数
from train import choose_device  # 复用训练脚本中的计算设备自动选择函数


PROJECT_ROOT = Path(__file__).resolve().parent  # 获取 app.py 所在的项目根目录
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_model.pth"  # 定义训练脚本默认生成的最佳模型路径
CANVAS_SIZE = 280  # 定义正方形绘图画布的像素边长
EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)/(\d+)")  # 编译用于提取当前训练轮次的正则表达式
ACCURACY_PATTERN = re.compile(r"最终测试准确率:\s*([0-9.]+)%")  # 编译用于提取最终测试准确率的正则表达式


class MnistApp:  # 封装 MNIST 图形界面的状态、布局和交互逻辑
    def __init__(self, root: tk.Tk) -> None:  # 初始化窗口、模型状态和全部界面控件
        self.root = root  # 保存 Tkinter 根窗口引用
        self.device = choose_device()  # 自动选择 CUDA、MPS 或 CPU 计算设备
        self.transform = build_transform()  # 创建与训练阶段一致的图像预处理流程
        self.model = None  # 缓存已经加载的推理模型，首次预测时再加载
        self.model_mtime = None  # 记录模型修改时间，以便训练后自动刷新缓存
        self.training_process = None  # 保存当前训练子进程，未训练时为 None
        self.training_thread = None  # 保存读取训练日志的后台线程
        self.log_queue = queue.Queue()  # 创建线程安全队列，用于传递训练日志
        self.reader_done = threading.Event()  # 创建事件标记，用于判断日志读取线程是否结束
        self.poll_job = None  # 保存界面轮询训练进程的定时任务编号
        self.stop_requested = False  # 记录用户是否主动停止了训练
        self.last_point = None  # 记录鼠标上一位置，用于连接连续笔画
        self.has_drawing = False  # 记录画布上是否已经有手写笔画
        self.drawing_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)  # 创建与画布同步的黑底灰度图像
        self.drawing_context = ImageDraw.Draw(self.drawing_image)  # 创建用于在灰度图像上画白色笔画的绘图对象
        self.candidate_variables = []  # 保存前三名候选概率对应的界面字符串变量
        self.candidate_bars = []  # 保存前三名候选概率对应的进度条控件
        self.root.title("MNIST 手写数字识别实验台")  # 设置窗口标题
        self.root.minsize(980, 720)  # 设置窗口允许缩放到的最小尺寸
        self.root.configure(background="#eef2f7")  # 设置窗口整体浅灰蓝色背景
        self._center_window(1120, 790)  # 把固定初始尺寸的窗口放到屏幕中央
        self._configure_style()  # 配置 Tkinter 主题和控件样式
        self._create_variables()  # 创建训练参数和结果展示所需的字符串变量
        self._build_layout()  # 创建左侧训练区和右侧推理区的全部控件
        self._reset_drawing()  # 初始化空画布和默认结果展示
        self._refresh_model_status()  # 显示当前是否存在可直接使用的模型
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)  # 注册窗口关闭时的资源清理函数
        self._append_log(f"界面已启动，当前计算设备：{self.device}")  # 在训练日志中记录启动状态

    def _center_window(self, width: int, height: int) -> None:  # 定义把窗口放到屏幕中央的工具函数
        screen_width = self.root.winfo_screenwidth()  # 读取当前屏幕宽度
        screen_height = self.root.winfo_screenheight()  # 读取当前屏幕高度
        left = max(0, (screen_width - width) // 2)  # 计算窗口水平方向的居中位置
        top = max(0, (screen_height - height) // 2)  # 计算窗口垂直方向的居中位置
        self.root.geometry(f"{width}x{height}+{left}+{top}")  # 应用窗口尺寸和居中坐标

    def _configure_style(self) -> None:  # 定义界面配色和控件样式
        style = ttk.Style(self.root)  # 创建当前窗口的 Tkinter 样式管理器
        if "clam" in style.theme_names():  # 检查系统是否提供更适合自定义颜色的 clam 主题
            style.theme_use("clam")  # 优先使用 clam 主题，使不同平台外观更一致
        style.configure("App.TFrame", background="#eef2f7")  # 设置页面背景样式
        style.configure("Card.TFrame", background="#ffffff")  # 设置白色卡片背景样式
        style.configure("Title.TLabel", background="#eef2f7", foreground="#172033", font=("TkDefaultFont", 20, "bold"))  # 设置窗口主标题样式
        style.configure("Subtitle.TLabel", background="#eef2f7", foreground="#667085", font=("TkDefaultFont", 10))  # 设置窗口副标题样式
        style.configure("Section.TLabel", background="#ffffff", foreground="#172033", font=("TkDefaultFont", 13, "bold"))  # 设置卡片标题样式
        style.configure("Field.TLabel", background="#ffffff", foreground="#344054", font=("TkDefaultFont", 10))  # 设置参数标签样式
        style.configure("Hint.TLabel", background="#ffffff", foreground="#667085", font=("TkDefaultFont", 9))  # 设置说明文字样式
        style.configure("Result.TLabel", background="#ffffff", foreground="#2563eb", font=("TkDefaultFont", 54, "bold"))  # 设置大号预测结果样式
        style.configure("Confidence.TLabel", background="#ffffff", foreground="#475467", font=("TkDefaultFont", 11))  # 设置置信度文字样式
        style.configure("Primary.TButton", font=("TkDefaultFont", 10, "bold"), padding=(14, 9))  # 设置主要按钮字体和内边距
        style.configure("Secondary.TButton", padding=(12, 8))  # 设置次要按钮内边距
        style.configure("Training.Horizontal.TProgressbar", troughcolor="#e4e7ec", background="#2563eb")  # 设置训练进度条颜色
        style.configure("Candidate.Horizontal.TProgressbar", troughcolor="#eaecf0", background="#10b981")  # 设置候选概率进度条颜色

    def _create_variables(self) -> None:  # 定义界面中会动态变化的字符串和数值变量
        self.epochs_var = tk.StringVar(value="5")  # 设置默认训练轮数
        self.batch_size_var = tk.StringVar(value="64")  # 设置默认批次大小
        self.learning_rate_var = tk.StringVar(value="0.001")  # 设置默认 Adam 学习率
        self.workers_var = tk.StringVar(value="2")  # 设置默认数据读取进程数
        self.seed_var = tk.StringVar(value="42")  # 设置默认随机种子
        self.target_var = tk.StringVar(value="98")  # 设置默认目标准确率百分比
        self.brush_var = tk.DoubleVar(value=18.0)  # 设置默认画笔粗细
        self.brush_text_var = tk.StringVar(value="18 px")  # 创建显示画笔粗细的文本变量
        self.prediction_var = tk.StringVar(value="?")  # 创建大号预测数字展示变量
        self.confidence_var = tk.StringVar(value="在画布上写一个数字，然后点击识别")  # 创建置信度提示变量
        self.model_status_var = tk.StringVar(value="正在检查模型...")  # 创建模型状态展示变量
        self.status_var = tk.StringVar(value="就绪")  # 创建底部运行状态展示变量
        self.training_detail_var = tk.StringVar(value="尚未开始训练")  # 创建训练进度文字变量
        for rank in range(3):  # 预先创建前三名候选结果的展示变量
            self.candidate_variables.append(tk.StringVar(value=f"第 {rank + 1} 名  --"))  # 添加一个候选结果文本变量

    def _build_layout(self) -> None:  # 定义窗口双栏布局和全部控件
        self.root.grid_rowconfigure(1, weight=1)  # 让窗口主体区域随窗口高度扩展
        self.root.grid_columnconfigure(0, weight=1)  # 让窗口内容占满水平方向
        header = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18, 24, 10))  # 创建顶部标题区域
        header.grid(row=0, column=0, sticky="ew")  # 把标题区域放到窗口第一行并填满宽度
        ttk.Label(header, text="MNIST 手写数字识别实验台", style="Title.TLabel").pack(anchor="w")  # 显示窗口主标题
        ttk.Label(header, text="调节 CNN 训练参数，并在右侧画布中直接测试识别效果", style="Subtitle.TLabel").pack(anchor="w", pady=(3, 0))  # 显示窗口用途说明
        body = ttk.Frame(self.root, style="App.TFrame", padding=(24, 8, 24, 18))  # 创建承载两个功能卡片的主体区域
        body.grid(row=1, column=0, sticky="nsew")  # 把主体区域放到标题下方并填满剩余空间
        body.grid_rowconfigure(0, weight=1)  # 让主体内容占满垂直方向
        body.grid_columnconfigure(0, weight=5, uniform="panels")  # 设置左侧训练卡片占据约一半宽度
        body.grid_columnconfigure(1, weight=5, uniform="panels")  # 设置右侧推理卡片占据约一半宽度
        self._build_training_panel(body)  # 创建左侧训练参数、日志和进度界面
        self._build_inference_panel(body)  # 创建右侧正方形画布和推理结果界面
        footer = ttk.Frame(self.root, style="App.TFrame", padding=(24, 0, 24, 12))  # 创建底部状态栏区域
        footer.grid(row=2, column=0, sticky="ew")  # 把状态栏放在窗口底部
        ttk.Label(footer, textvariable=self.status_var, style="Subtitle.TLabel").pack(anchor="w")  # 显示当前操作状态

    def _build_training_panel(self, parent: ttk.Frame) -> None:  # 定义左侧训练控制面板
        panel = tk.Frame(parent, background="#ffffff", highlightbackground="#d0d5dd", highlightthickness=1)  # 创建带浅灰边框的白色训练卡片
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 9))  # 把训练卡片放在左栏并保留卡片间距
        panel.grid_columnconfigure(0, weight=1)  # 让训练卡片内部内容填满水平方向
        content = ttk.Frame(panel, style="Card.TFrame", padding=(20, 18, 20, 18))  # 创建训练卡片内部内容区域
        content.grid(row=0, column=0, sticky="nsew")  # 让内容区域填满整个训练卡片
        content.grid_columnconfigure(1, weight=1)  # 让参数输入框随卡片宽度扩展
        ttk.Label(content, text="训练参数", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))  # 显示训练参数标题
        self._add_parameter_row(content, 1, "训练轮数", self.epochs_var, from_=1, to=50)  # 添加训练轮数输入控件
        self._add_parameter_row(content, 2, "批次大小", self.batch_size_var, from_=1, to=1024)  # 添加批次大小输入控件
        self._add_parameter_row(content, 3, "学习率", self.learning_rate_var)  # 添加学习率文本输入控件
        self._add_parameter_row(content, 4, "读取进程", self.workers_var, from_=0, to=16)  # 添加 DataLoader 进程数输入控件
        self._add_parameter_row(content, 5, "随机种子", self.seed_var, from_=0, to=999999)  # 添加随机种子输入控件
        self._add_parameter_row(content, 6, "目标准确率 (%)", self.target_var)  # 添加目标准确率输入控件
        ttk.Label(content, text="数据目录：data/    输出目录：outputs/", style="Hint.TLabel").grid(row=7, column=0, columnspan=2, sticky="w", pady=(2, 10))  # 显示固定数据与输出目录
        button_row = ttk.Frame(content, style="Card.TFrame")  # 创建训练操作按钮所在行
        button_row.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 12))  # 把按钮行放在参数区域下方
        button_row.grid_columnconfigure(0, weight=1)  # 让开始训练按钮占据主要宽度
        self.start_button = ttk.Button(button_row, text="开始训练", style="Primary.TButton", command=self._start_training)  # 创建启动后台训练的按钮
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))  # 把开始训练按钮放到左侧
        self.stop_button = ttk.Button(button_row, text="停止", style="Secondary.TButton", command=self._stop_training, state="disabled")  # 创建训练期间使用的停止按钮
        self.stop_button.grid(row=0, column=1, sticky="e")  # 把停止按钮放到右侧
        self.training_progress = ttk.Progressbar(content, style="Training.Horizontal.TProgressbar", mode="determinate", maximum=5, value=0)  # 创建按 epoch 显示的确定性训练进度条
        self.training_progress.grid(row=9, column=0, columnspan=2, sticky="ew")  # 把进度条放在按钮下方并填满宽度
        ttk.Label(content, textvariable=self.training_detail_var, style="Hint.TLabel").grid(row=10, column=0, columnspan=2, sticky="w", pady=(4, 10))  # 显示当前训练轮次和完成状态
        ttk.Label(content, text="训练日志", style="Field.TLabel").grid(row=11, column=0, columnspan=2, sticky="w", pady=(0, 5))  # 显示训练日志标题
        log_frame = tk.Frame(content, background="#ffffff")  # 创建用于容纳日志文本和滚动条的框架
        log_frame.grid(row=12, column=0, columnspan=2, sticky="nsew")  # 让日志框架填满剩余空间
        content.grid_rowconfigure(12, weight=1)  # 让日志框架占用参数下方所有剩余高度
        log_frame.grid_rowconfigure(0, weight=1)  # 让日志文本填满日志框架高度
        log_frame.grid_columnconfigure(0, weight=1)  # 让日志文本填满日志框架宽度
        self.log_text = tk.Text(log_frame, height=12, wrap="word", state="disabled", background="#101828", foreground="#d1fadf", insertbackground="#ffffff", relief="flat", padx=10, pady=8, font=("TkFixedFont", 9))  # 创建深色只读训练日志区域
        self.log_text.grid(row=0, column=0, sticky="nsew")  # 把日志文本放到日志框架中
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)  # 创建用于滚动训练日志的垂直滚动条
        log_scrollbar.grid(row=0, column=1, sticky="ns")  # 把滚动条放到日志文本右侧
        self.log_text.configure(yscrollcommand=log_scrollbar.set)  # 让日志文本滚动位置和滚动条双向同步

    def _add_parameter_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, from_=None, to=None) -> None:  # 定义快速添加一行参数标签和输入框的方法
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 14), pady=4)  # 在参数行左侧显示参数名称
        if from_ is not None and to is not None:  # 判断当前参数是否需要数字步进框
            widget = ttk.Spinbox(parent, textvariable=variable, from_=from_, to=to, increment=1, width=18)  # 创建带范围限制的数字步进框
        else:  # 处理学习率等适合直接文本输入的参数
            widget = ttk.Entry(parent, textvariable=variable, width=20)  # 创建接受文本的输入框
        widget.grid(row=row, column=1, sticky="ew", pady=4)  # 把参数输入框放到参数名称右侧并填满剩余宽度

    def _build_inference_panel(self, parent: ttk.Frame) -> None:  # 定义右侧绘图和推理面板
        panel = tk.Frame(parent, background="#ffffff", highlightbackground="#d0d5dd", highlightthickness=1)  # 创建带浅灰边框的白色推理卡片
        panel.grid(row=0, column=1, sticky="nsew", padx=(9, 0))  # 把推理卡片放在右栏并保留卡片间距
        panel.grid_columnconfigure(0, weight=1)  # 让推理卡片内容充满水平方向
        panel.grid_rowconfigure(1, weight=1)  # 让画布区域占用主要垂直空间
        content = ttk.Frame(panel, style="Card.TFrame", padding=(20, 18, 20, 18))  # 创建推理卡片内部内容区域
        content.grid(row=0, column=0, sticky="nsew")  # 让内容区域填满推理卡片
        content.grid_columnconfigure(0, weight=1)  # 让画布和结果区域在水平方向居中扩展
        ttk.Label(content, text="手写数字识别", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 3))  # 显示推理区域标题
        ttk.Label(content, text="请用黑底白字书写，笔画尽量位于画布中央", style="Hint.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 12))  # 显示画布书写提示
        canvas_shell = tk.Frame(content, background="#101828", padx=8, pady=8)  # 创建围绕画布的深色边框容器
        canvas_shell.grid(row=2, column=0)  # 把画布容器放在推理卡片中央
        self.canvas = tk.Canvas(canvas_shell, width=CANVAS_SIZE, height=CANVAS_SIZE, background="#000000", highlightthickness=0, cursor="crosshair")  # 创建固定 280×280 的黑色正方形画布
        self.canvas.pack()  # 把画布放入深色边框容器
        self.canvas.bind("<Button-1>", self._begin_stroke)  # 在按下鼠标左键时开始新笔画
        self.canvas.bind("<B1-Motion>", self._continue_stroke)  # 在按住鼠标移动时连续绘制笔画
        self.canvas.bind("<ButtonRelease-1>", self._end_stroke)  # 在释放鼠标时结束当前笔画
        canvas_controls = ttk.Frame(content, style="Card.TFrame")  # 创建画笔控制行
        canvas_controls.grid(row=3, column=0, sticky="ew", pady=(12, 10))  # 把画笔控制行放在画布下方
        canvas_controls.grid_columnconfigure(2, weight=1)  # 让画笔滑块占据剩余水平空间
        ttk.Label(canvas_controls, text="画笔", style="Field.TLabel").grid(row=0, column=0, padx=(0, 8))  # 显示画笔标签
        ttk.Scale(canvas_controls, variable=self.brush_var, from_=8, to=36, command=self._update_brush_text).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(0, 8))  # 创建调节笔画粗细的滑块
        ttk.Label(canvas_controls, textvariable=self.brush_text_var, style="Hint.TLabel", width=7).grid(row=0, column=3)  # 显示当前画笔粗细数值
        clear_button = ttk.Button(canvas_controls, text="清空画布", style="Secondary.TButton", command=self._reset_drawing)  # 创建清空画布按钮
        clear_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0), padx=(0, 8))  # 把清空按钮放在控制行第二行
        predict_button = ttk.Button(canvas_controls, text="识别数字", style="Primary.TButton", command=self._predict_drawing)  # 创建对当前画布执行 inference 的按钮
        predict_button.grid(row=1, column=2, columnspan=2, sticky="ew", pady=(10, 0))  # 把识别按钮放在清空按钮右侧
        result_shell = ttk.Frame(content, style="Card.TFrame")  # 创建预测结果区域
        result_shell.grid(row=4, column=0, sticky="ew", pady=(2, 8))  # 把预测结果区域放在画笔控制下方
        result_shell.grid_columnconfigure(1, weight=1)  # 让结果区域中的候选进度条扩展
        ttk.Label(result_shell, textvariable=self.prediction_var, style="Result.TLabel").grid(row=0, column=0, rowspan=2, padx=(12, 24))  # 用大号蓝色字体显示预测数字
        ttk.Label(result_shell, text="预测结果", style="Field.TLabel").grid(row=0, column=1, sticky="sw")  # 显示预测结果说明
        ttk.Label(result_shell, textvariable=self.confidence_var, style="Confidence.TLabel").grid(row=1, column=1, sticky="nw", pady=(2, 0))  # 显示预测置信度或操作提示
        candidates = ttk.Frame(content, style="Card.TFrame")  # 创建前三名候选概率列表区域
        candidates.grid(row=5, column=0, sticky="ew", pady=(6, 4))  # 把候选列表放在预测结果下方
        candidates.grid_columnconfigure(1, weight=1)  # 让每个候选进度条填满可用宽度
        for rank in range(3):  # 创建三行候选结果，分别对应第 1 到第 3 名
            ttk.Label(candidates, textvariable=self.candidate_variables[rank], style="Field.TLabel", width=15).grid(row=rank, column=0, sticky="w", pady=3)  # 显示当前候选排名、数字和概率
            bar = ttk.Progressbar(candidates, style="Candidate.Horizontal.TProgressbar", mode="determinate", maximum=100, value=0)  # 创建表示候选概率的进度条
            bar.grid(row=rank, column=1, sticky="ew", padx=(8, 0), pady=3)  # 把候选概率条放到文本右侧
            self.candidate_bars.append(bar)  # 保存进度条控件，以便预测后更新数值
        ttk.Label(content, textvariable=self.model_status_var, style="Hint.TLabel").grid(row=6, column=0, sticky="w", pady=(10, 0))  # 显示当前使用的模型和验证准确率

    def _refresh_model_status(self) -> None:  # 定义刷新模型文件状态的显示信息
        if MODEL_PATH.exists():  # 判断训练产物中的最佳模型是否存在
            self.model_status_var.set(f"模型：outputs/best_model.pth（{MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB）")  # 显示模型路径和文件大小
        else:  # 处理模型尚未训练的情况
            self.model_status_var.set("模型：尚未生成，请先在左侧开始训练")  # 提示用户先完成训练

    def _update_brush_text(self, value) -> None:  # 定义滑块数值变化时更新画笔粗细文字的方法
        width = float(value)  # 把 Tkinter 传入的滑块值转换为浮点数
        self.brush_text_var.set(f"{width:.0f} px")  # 用整数像素显示当前画笔粗细

    def _begin_stroke(self, event: tk.Event) -> None:  # 定义鼠标按下时开始绘制笔画的函数
        self.last_point = (event.x, event.y)  # 记录当前笔画起始坐标
        self.has_drawing = True  # 标记画布已经包含手写内容
        self.prediction_var.set("?")  # 开始新数字时清除旧预测数字
        self.confidence_var.set("正在书写，完成后点击“识别数字”")  # 把结果说明切换为书写提示
        for bar in self.candidate_bars:  # 遍历所有候选概率进度条
            bar.configure(value=0)  # 清除上一次预测的候选概率
        self._paint_segment(event.x, event.y, event.x, event.y)  # 在点击位置绘制一个圆形起始笔画

    def _continue_stroke(self, event: tk.Event) -> None:  # 定义鼠标拖动时延续笔画的函数
        if self.last_point is None:  # 检查是否缺少上一个坐标
            return  # 没有起点时不绘制，避免产生无意义的线段
        start_x, start_y = self.last_point  # 读取上一位置的横纵坐标
        self._paint_segment(start_x, start_y, event.x, event.y)  # 在上一位置和当前位置之间绘制线段
        self.last_point = (event.x, event.y)  # 更新位置，供下一次拖动继续绘制

    def _end_stroke(self, event: tk.Event) -> None:  # 定义鼠标释放时结束当前笔画的函数
        self.last_point = None  # 清除上一位置，避免下一次点击时错误连接两条笔画

    def _paint_segment(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:  # 定义同步绘制 Tkinter 线段和 Pillow 图像的函数
        brush_width = max(1, int(self.brush_var.get()))  # 读取画笔粗细并确保至少为 1 像素
        self.canvas.create_line(start_x, start_y, end_x, end_y, fill="#ffffff", width=brush_width, capstyle=tk.ROUND, smooth=True, splinesteps=12)  # 在可见画布上绘制圆头白色线段
        radius = brush_width / 2  # 计算圆形笔尖半径
        self.canvas.create_oval(start_x - radius, start_y - radius, start_x + radius, start_y + radius, fill="#ffffff", outline="#ffffff")  # 在可见画布起点补充圆点，避免短笔画出现缺口
        self.drawing_context.line((start_x, start_y, end_x, end_y), fill=255, width=brush_width, joint="curve")  # 在灰度图像中的同一位置绘制白色线段
        self.drawing_context.ellipse((start_x - radius, start_y - radius, start_x + radius, start_y + radius), fill=255)  # 在灰度图像中绘制圆头笔尖

    def _reset_drawing(self) -> None:  # 定义清空画布和预测结果的方法
        self.canvas.delete("all")  # 删除可见画布中的所有笔画
        self.drawing_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)  # 创建新的全黑灰度画布
        self.drawing_context = ImageDraw.Draw(self.drawing_image)  # 重新绑定用于绘制灰度图像的上下文
        self.last_point = None  # 清除尚未结束的笔画位置
        self.has_drawing = False  # 标记当前画布为空
        self.prediction_var.set("?")  # 把大号结果数字恢复为问号
        self.confidence_var.set("在画布上写一个数字，然后点击识别")  # 恢复操作提示
        for rank in range(3):  # 遍历三个候选结果
            self.candidate_variables[rank].set(f"第 {rank + 1} 名  --")  # 清除候选数字和概率文字
            self.candidate_bars[rank].configure(value=0)  # 把候选概率进度条清零
        self.status_var.set("画布已清空")  # 在底部状态栏显示清空结果
        self.canvas.focus_set()  # 把键盘和绘图焦点重新交给画布

    def _get_inference_model(self):  # 定义按需加载并缓存推理模型的方法
        if not MODEL_PATH.exists():  # 检查默认模型文件是否已经由训练生成
            raise FileNotFoundError("尚未生成模型，请先完成一次训练")  # 不存在模型时给出界面可展示的错误信息
        current_mtime = MODEL_PATH.stat().st_mtime  # 读取模型文件的最后修改时间
        if self.model is None or self.model_mtime != current_mtime:  # 判断模型是否尚未加载或已经被训练更新
            self.model, checkpoint = load_model(MODEL_PATH, self.device)  # 加载模型参数并切换到评估模式
            self.model_mtime = current_mtime  # 记录当前加载模型对应的修改时间
            self.model_status_var.set(f"模型：best_model.pth，验证准确率 {checkpoint['validation_accuracy']:.2%}")  # 在界面中显示模型的验证准确率
        return self.model  # 返回可以直接用于推理的缓存模型

    def _predict_drawing(self) -> None:  # 定义读取正方形画布并执行数字识别的方法
        if not self.has_drawing:  # 检查画布是否为空
            messagebox.showwarning("画布为空", "请先在画布上书写一个数字。")  # 提示用户先绘制内容
            return  # 空画布不执行推理
        try:  # 捕获模型加载或推理过程中可能出现的异常
            self.status_var.set("正在识别手写数字...")  # 更新底部状态提示
            self.root.update_idletasks()  # 立即刷新界面上的状态文字
            model = self._get_inference_model()  # 获取最新训练得到的模型
            tensor = preprocess_pil_image(self.drawing_image, self.transform, invert=False)  # 把画布图像转换为标准化模型输入
            result = predict_tensors(model, [tensor], self.device, top_k=3)[0]  # 执行推理并取出第一张图片的最终结果
            self.prediction_var.set(str(result["prediction"]))  # 用大号数字显示预测类别
            self.confidence_var.set(f"置信度 {result['confidence']:.2%}")  # 显示最高候选类别的置信度
            for rank, candidate in enumerate(result["top_k"]):  # 依次更新前三名候选结果
                probability_percent = candidate["probability"] * 100  # 把 0 到 1 的概率换算为百分比
                self.candidate_variables[rank].set(f"第 {rank + 1} 名  {candidate['digit']}  {candidate['probability']:.2%}")  # 显示候选排名、数字和概率
                self.candidate_bars[rank].configure(value=probability_percent)  # 用进度条长度直观展示候选概率
            self.status_var.set(f"识别完成：数字 {result['prediction']}")  # 在底部状态栏显示最终预测
        except Exception as error:  # 捕获文件、模型或图像处理相关的异常
            self.status_var.set("识别失败")  # 更新底部状态为失败
            messagebox.showerror("识别失败", str(error))  # 使用消息框显示可读错误信息

    def _read_training_config(self):  # 定义读取并校验训练界面参数的方法
        try:  # 捕获数字格式或范围错误
            epochs = int(self.epochs_var.get())  # 把训练轮数文本转换为整数
            batch_size = int(self.batch_size_var.get())  # 把批次大小文本转换为整数
            learning_rate = float(self.learning_rate_var.get())  # 把学习率文本转换为浮点数
            workers = int(self.workers_var.get())  # 把数据读取进程数转换为整数
            seed = int(self.seed_var.get())  # 把随机种子转换为整数
            target_percent = float(self.target_var.get())  # 把目标准确率百分比转换为浮点数
        except ValueError as error:  # 捕获用户输入非数字的情况
            raise ValueError("训练参数必须是有效数字，请检查输入。") from error  # 把底层格式错误转换为界面提示
        if epochs < 1:  # 检查训练轮数是否为正数
            raise ValueError("训练轮数必须大于等于 1。")  # 拒绝无效训练轮数
        if batch_size < 1:  # 检查批次大小是否为正数
            raise ValueError("批次大小必须大于等于 1。")  # 拒绝无效批次大小
        if learning_rate <= 0:  # 检查学习率是否为正数
            raise ValueError("学习率必须大于 0。")  # 拒绝无效学习率
        if workers < 0:  # 检查数据读取进程数是否非负
            raise ValueError("读取进程数不能小于 0。")  # 拒绝无效进程数
        if not 0 < target_percent <= 100:  # 检查目标准确率是否在百分比有效范围内
            raise ValueError("目标准确率必须大于 0 且不超过 100。")  # 拒绝超出范围的准确率
        return epochs, batch_size, learning_rate, workers, seed, target_percent / 100  # 返回整理后的全部训练参数

    def _start_training(self) -> None:  # 定义启动后台训练子进程的方法
        if self.training_process is not None and self.training_process.poll() is None:  # 防止同时启动多个训练进程
            messagebox.showinfo("训练进行中", "请等待当前训练结束或先点击停止。")  # 提示当前已有训练任务
            return  # 不重复启动训练
        try:  # 捕获训练参数校验错误
            epochs, batch_size, learning_rate, workers, seed, target_accuracy = self._read_training_config()  # 读取并验证界面训练参数
        except ValueError as error:  # 捕获非法参数
            messagebox.showerror("参数错误", str(error))  # 显示具体参数错误
            return  # 参数无效时不启动训练
        command = [  # 创建不经过 shell 的训练子进程命令列表
            sys.executable,  # 使用当前 Python 解释器，确保能够导入 PyTorch
            "-u",  # 开启无缓冲输出，使训练日志实时显示在界面中
            str(PROJECT_ROOT / "train.py"),  # 指定项目训练脚本的完整路径
            "--epochs", str(epochs),  # 传入界面设置的训练轮数
            "--batch-size", str(batch_size),  # 传入界面设置的批次大小
            "--learning-rate", str(learning_rate),  # 传入界面设置的学习率
            "--workers", str(workers),  # 传入界面设置的数据读取进程数
            "--seed", str(seed),  # 传入界面设置的随机种子
            "--target-accuracy", f"{target_accuracy:.6f}",  # 传入界面设置的最终目标准确率
        ]  # 完成训练命令构造
        try:  # 捕获子进程启动失败
            self.training_process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)  # 启动训练并合并捕获标准输出与错误输出
        except OSError as error:  # 捕获解释器或脚本无法启动的错误
            messagebox.showerror("启动失败", str(error))  # 向用户显示启动失败原因
            return  # 子进程未启动时不继续更新训练状态
        self.stop_requested = False  # 清除上一次停止请求标记
        self.reader_done.clear()  # 清除日志线程完成标记
        self.training_progress.configure(maximum=epochs, value=0)  # 根据训练轮数重置训练进度条
        self.training_detail_var.set(f"正在准备训练：0/{epochs} 轮")  # 显示训练准备状态
        self.status_var.set("训练进程已启动")  # 更新底部状态
        self._set_training_state(True)  # 切换按钮和状态到训练中
        self._append_log("开始训练：" + " ".join(command[1:]))  # 在日志中显示本次训练命令
        self.training_thread = threading.Thread(target=self._read_training_output, daemon=True)  # 创建持续读取训练输出的守护线程
        self.training_thread.start()  # 启动日志读取线程
        self._poll_training()  # 开始周期性检查训练进程状态

    def _read_training_output(self) -> None:  # 定义在后台线程中读取训练进程输出的方法
        try:  # 确保读取结束时一定设置线程完成标记
            if self.training_process is None or self.training_process.stdout is None:  # 检查训练进程和输出管道是否有效
                return  # 无法读取时直接结束线程
            for line in self.training_process.stdout:  # 逐行读取训练脚本输出
                self.log_queue.put(line.rstrip())  # 把去除末尾换行的日志放入线程安全队列
        finally:  # 无论正常结束还是异常都执行收尾
            self.reader_done.set()  # 标记日志读取线程已经结束

    def _poll_training(self) -> None:  # 定义在 Tkinter 主线程中检查训练状态的函数
        self._drain_log_queue()  # 把后台线程收集到的日志更新到界面
        if self.training_process is None:  # 检查训练进程是否已经被清理
            return  # 没有训练任务时停止轮询
        return_code = self.training_process.poll()  # 读取子进程退出码，运行中时返回 None
        if return_code is None or not self.reader_done.is_set():  # 判断训练仍未结束或日志尚未读完
            self.poll_job = self.root.after(150, self._poll_training)  # 150 毫秒后再次检查，保持界面响应
            return  # 本次检查结束，等待下一次轮询
        self._drain_log_queue()  # 最后再读取一次队列中剩余的日志
        self.poll_job = None  # 清除已完成的定时任务编号
        self._finish_training(return_code)  # 根据退出码处理训练完成或失败

    def _drain_log_queue(self) -> None:  # 定义把线程队列中的训练日志写入文本框的方法
        while True:  # 持续读取队列直到暂时没有新消息
            try:  # 尝试从队列中取出一条日志
                line = self.log_queue.get_nowait()  # 非阻塞读取一条日志，队列为空时立即抛出异常
            except queue.Empty:  # 处理当前没有新日志的情况
                break  # 结束本次批量更新
            self._append_log(line)  # 把日志追加到只读文本框
            epoch_match = EPOCH_PATTERN.search(line)  # 检查日志行是否包含训练轮次信息
            if epoch_match:  # 判断是否需要更新训练进度
                current_epoch = int(epoch_match.group(1))  # 提取当前已经完成的轮次
                total_epochs = int(epoch_match.group(2))  # 提取本次训练计划的总轮数
                self.training_progress.configure(maximum=total_epochs, value=current_epoch)  # 同步更新进度条的最大值和当前值
                self.training_detail_var.set(f"训练进度：{current_epoch}/{total_epochs} 轮")  # 更新文字进度
            accuracy_match = ACCURACY_PATTERN.search(line)  # 检查日志是否包含最终测试准确率
            if accuracy_match:  # 判断是否已经输出最终准确率
                self.status_var.set(f"训练完成，最终测试准确率 {accuracy_match.group(1)}%")  # 在底部状态栏显示最终结果

    def _finish_training(self, return_code: int) -> None:  # 定义训练子进程结束后的收尾逻辑
        self._set_training_state(False)  # 恢复开始按钮并禁用停止按钮
        self.training_process = None  # 清除已经结束的子进程引用
        self.training_thread = None  # 清除已经结束的日志线程引用
        self.model = None  # 丢弃旧模型缓存，确保下次推理加载刚训练的新模型
        self.model_mtime = None  # 清除旧模型修改时间
        self._refresh_model_status()  # 刷新模型文件状态
        if self.stop_requested:  # 判断训练是否由用户主动停止
            self.training_detail_var.set("训练已停止")  # 在进度区域显示停止状态
            self.status_var.set("训练已由用户停止")  # 在底部状态栏显示停止状态
            self.stop_requested = False  # 清除停止请求标记
            return  # 主动停止后不再检查训练指标
        if return_code == 0:  # 判断训练脚本是否正常完成并达到目标准确率
            metrics_path = PROJECT_ROOT / "outputs" / "metrics.json"  # 定位训练脚本生成的最终指标文件
            try:  # 捕获指标文件缺失或格式错误
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))  # 读取最终测试指标 JSON 文件
                accuracy = metrics["test_accuracy"]  # 提取最终测试准确率
                self.training_detail_var.set(f"训练完成：测试准确率 {accuracy:.2%}")  # 在进度区域显示最终准确率
                self.status_var.set("训练完成，可以开始手写识别")  # 更新底部状态
                messagebox.showinfo("训练完成", f"模型已更新。\n最终测试准确率：{accuracy:.2%}")  # 弹窗通知训练成功
            except (OSError, KeyError, json.JSONDecodeError) as error:  # 捕获无法读取训练指标的情况
                self.training_detail_var.set("训练完成，但无法读取指标文件")  # 显示指标读取失败状态
                messagebox.showwarning("训练完成", f"训练已结束，但读取指标失败：\n{error}")  # 提示指标文件异常
            return  # 成功完成训练后结束收尾
        self.training_detail_var.set(f"训练结束，退出码 {return_code}")  # 显示训练异常结束的退出码
        self.status_var.set("训练未达到目标或发生错误，请查看日志")  # 更新底部错误提示
        messagebox.showerror("训练失败", "训练没有正常完成，请查看左侧训练日志。")  # 弹窗提示用户检查日志

    def _stop_training(self) -> None:  # 定义由界面停止当前训练任务的方法
        if self.training_process is None or self.training_process.poll() is not None:  # 检查当前是否存在正在运行的训练任务
            return  # 没有活动任务时不执行停止操作
        self.stop_requested = True  # 记录用户主动停止训练
        self.training_process.terminate()  # 向训练子进程发送终止信号
        self.training_detail_var.set("正在停止训练...")  # 显示正在停止的状态
        self.status_var.set("已请求停止训练")  # 更新底部状态

    def _set_training_state(self, running: bool) -> None:  # 定义训练期间按钮的启用状态
        if running:  # 判断当前是否正在训练
            self.start_button.configure(state="disabled")  # 训练期间禁用开始按钮，避免重复启动
            self.stop_button.configure(state="normal")  # 训练期间启用停止按钮
        else:  # 处理训练结束后的状态恢复
            self.start_button.configure(state="normal")  # 恢复开始训练按钮
            self.stop_button.configure(state="disabled")  # 禁用不再需要的停止按钮

    def _append_log(self, message: str) -> None:  # 定义向只读日志文本框追加一行文字的方法
        self.log_text.configure(state="normal")  # 临时解除只读状态，以便插入新日志
        self.log_text.insert(tk.END, message + "\n")  # 在文本末尾插入日志和换行
        self.log_text.see(tk.END)  # 自动滚动到最新一行
        self.log_text.configure(state="disabled")  # 恢复只读状态，防止用户误改日志

    def _on_close(self) -> None:  # 定义关闭窗口前清理训练进程和定时任务的函数
        if self.training_process is not None and self.training_process.poll() is None:  # 检查是否仍有训练子进程在运行
            self.training_process.terminate()  # 关闭界面时终止未完成的训练
        if self.poll_job is not None:  # 检查是否还有等待执行的界面轮询任务
            self.root.after_cancel(self.poll_job)  # 取消定时任务，避免窗口销毁后回调报错
        self.root.destroy()  # 销毁 Tkinter 窗口并退出主循环


if __name__ == "__main__":  # 只有直接运行本文件时才创建图形界面
    root = tk.Tk()  # 创建 Tkinter 根窗口
    app = MnistApp(root)  # 创建并初始化 MNIST 图形界面应用
    root.mainloop()  # 启动 Tkinter 事件循环，等待用户操作
