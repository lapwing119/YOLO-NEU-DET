from pathlib import Path
from uuid import uuid4

import torch
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


# 项目根目录，也就是 app.py 所在目录
PROJECT_DIR = Path(__file__).resolve().parent

# 训练完成的最佳模型权重
MODEL_PATH = (
    PROJECT_DIR
    / "weights"
    / "best.pt"
)

# 保存检测结果图的目录
RESULT_DIR = PROJECT_DIR / "static" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 限制上传图片大小为 10 MB
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 有 NVIDIA CUDA 时使用 GPU 0；否则使用 CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# 启动程序时加载一次模型，后续上传图片都复用它
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"找不到模型权重文件：{MODEL_PATH}")

print(f"正在加载模型：{MODEL_PATH}")
print(f"推理设备：{DEVICE}")
model = YOLO(str(MODEL_PATH))
print("模型加载完成。")


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result_image = None
    detections = []
    confidence = 0.25

    if request.method == "POST":
        # 读取网页上传的置信度阈值
        try:
            confidence = float(request.form.get("confidence", "0.25"))
        except ValueError:
            confidence = 0.25

        # 限制阈值范围，防止收到不合理的输入
        confidence = max(0.05, min(0.90, confidence))

        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "请先选择一张图片。"
        else:
            suffix = Path(uploaded_file.filename).suffix.lower()

            if suffix not in ALLOWED_EXTENSIONS:
                error = "图片格式不支持，请选择 JPG、JPEG、PNG 或 BMP 图片。"
            else:
                try:
                    # 读取上传图片并转换为 RGB 格式
                    image = Image.open(uploaded_file.stream).convert("RGB")

                    # 调用 YOLO 模型进行预测
                    result = model.predict(
                        source=image,
                        conf=confidence,
                        device=DEVICE,
                        verbose=False,
                    )[0]

                    # 画出预测框。Ultralytics 返回的图像通道顺序是 BGR，
                    # 转成 RGB 后再用 Pillow 保存。
                    annotated_bgr = result.plot()
                    annotated_rgb = Image.fromarray(
                        annotated_bgr[:, :, ::-1].copy()
                    )

                    # 使用随机文件名，避免不同上传结果互相覆盖
                    result_image = f"{uuid4().hex}.jpg"
                    annotated_rgb.save(RESULT_DIR / result_image, format="JPEG")

                    # 整理每个检测框的信息，供网页表格显示
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        class_name = result.names[class_id]
                        score = float(box.conf[0].item())
                        coordinates = [
                            round(value)
                            for value in box.xyxy[0].tolist()
                        ]

                        detections.append(
                            {
                                "class_name": class_name,
                                "confidence": score,
                                "box": ", ".join(str(value) for value in coordinates),
                            }
                        )

                except (UnidentifiedImageError, OSError):
                    error = "无法读取这张图片，请确认文件没有损坏。"
                except Exception:
                    app.logger.exception("图片预测失败")
                    error = "图片预测失败，请检查 PowerShell 窗口中的错误信息。"

    return render_template(
        "index.html",
        error=error,
        result_image=result_image,
        detections=detections,
        confidence=confidence,
    )


@app.errorhandler(413)
def file_too_large(error):
    return (
        render_template(
            "index.html",
            error="图片超过 10 MB，请选择较小的图片。",
            result_image=None,
            detections=[],
            confidence=0.25,
        ),
        413,
    )


if __name__ == "__main__":
    # 只允许本机访问，适合当前的本地演示
    app.run(host="127.0.0.1", port=5000, debug=False)