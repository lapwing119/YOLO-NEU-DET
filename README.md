# NEU-DET 钢材表面缺陷检测项目

本项目使用 YOLO26n 对 NEU-DET 钢材表面缺陷数据集进行目标检测。程序提供网页界面，用户上传图片后，可以查看模型预测的缺陷位置、类别和置信度。

## 检测类别

模型识别以下 6 类缺陷：

- crazing（龟裂）
- inclusion（夹杂）
- patches（斑块）
- pitted_surface（麻点）
- rolled-in_scale（氧化铁皮）
- scratches（划痕）

## 项目结构

```text
YOLO-NEU-DET/
├── app.py                  # Flask 图片检测程序
├── prepare_dataset.py      # 将 XML 标注转换成 YOLO 格式并划分数据集
├── requirements.txt        # Python 项目依赖
├── templates/
│   └── index.html          # 上传和展示结果的网页
├── weights/
│   └── best.pt             # 训练得到的最佳模型权重
├── dataset/
│   └── data.yaml           # YOLO 数据集配置
└── .gitignore

原始数据、转换后的图片和标签、训练记录以及临时预测结果不包含在 Git 版本库中。请根据下方说明准备数据集。

## 环境要求
- Windows 10 或 Windows 11
- Python 3.11
- NVIDIA 显卡可用于 CUDA 推理；没有 NVIDIA 显卡时可使用 CPU，但速度会较慢

## 安装
在项目根目录打开 PowerShell，创建并启用虚拟环境：

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

先根据电脑的显卡和 CUDA 情况，从 PyTorch 官方安装页面选择 Windows、pip 和合适的 CUDA 版本，安装 PyTorch：
https://pytorch.org/get-started/locally/
然后安装项目依赖：

python -m pip install -r requirements.txt

本项目开发和验证时使用 Python 3.11.9、PyTorch 2.12.1+cu132、Ultralytics 8.4.182、Flask 3.1.3 和 NVIDIA GeForce RTX 4050 Laptop GPU。其他电脑的 PyTorch/CUDA 安装命令可能不同。

## 启动网页检测程序

在项目根目录打开 PowerShell，运行：

.\.venv\Scripts\python.exe .\app.py

看到 Flask 启动信息后，在浏览器打开：
http://127.0.0.1:5000
选择一张 JPG、JPEG、PNG 或 BMP 图片并开始检测。网页会显示带有检测框的结果图片，以及每个检测目标的类别、置信度和边界框坐标。
停止服务时，在运行 Flask 的终端按 Ctrl+C。

## 准备数据集

从指定来源获取 NEU-DET 数据集，并按数据集许可要求使用。
将压缩包放入 downloads 文件夹并解压到 raw 文件夹，使原始数据位于 raw/NEU-DET。
在项目根目录运行：

.\.venv\Scripts\python.exe .\prepare_dataset.py

脚本会读取 XML 标注，转换为 YOLO 检测标签，并生成训练集、验证集、测试集及 dataset/data.yaml。数据集图片和标签不包含在本仓库中，需要自行准备。

## 训练模型

数据集准备好且 PyTorch/CUDA 环境已配置后，可以使用 Ultralytics 训练命令。训练权重、数据配置路径和 GPU 设置需根据本机实际情况确认。当前交付使用的模型权重是 weights/best.pt。

## 测试结果

独立测试集包含 180 张图片。训练完成后，在独立测试集上得到的主要指标为：
- Precision：0.719
- Recall：0.674
- mAP@0.5：0.739
- mAP@0.5:0.95：0.401
按类别的 AP@0.5 和完整测试说明请查看 模型测试结果.md。

## 注意事项

上传图片由本机运行的 Flask 程序处理；使用地址是 127.0.0.1，仅供本机访问。
weights/best.pt 是本项目训练得到的权重；yolo26n.pt 是训练时使用的初始预训练权重，两者用途不同。
本仓库不包含原始数据集、转换后的图片和标签、虚拟环境或训练过程文件。


