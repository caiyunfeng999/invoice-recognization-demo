# Invoice OCR Backend

后端采用 `main.py + invoice_app 子功能模块` 的结构。`main.py` 只负责启动 Flask，具体功能全部拆分到 `invoice_app/` 目录中。

## 启动

```bash
conda activate pyhon11-opencv
cd ~/invoice_ocr_web/backend
pip install -r requirements.txt
python main.py
```

兼容旧命令：

```bash
python app.py
```

API 地址：

```text
http://127.0.0.1:5001
```

## API

- `GET /api/health`：后端健康检查
- `GET /api/methods`：获取可用图像处理方法
- `GET /api/detectors/status`：获取 YOLOv8n、Faster R-CNN、D-FINE-L 的权重和运行时状态
- `POST /api/process/<method>`：执行图像处理并返回 base64 图像
- `POST /api/ocr`：执行 PDF 文本直读、PaddleOCR 或 Tesseract OCR，并返回结构化字段
- `POST /api/yolo/detect`：字段检测预览接口，兼容旧 YOLO 路径；可通过 `detector=yolo|faster_rcnn|dfine_l` 选择检测模型

## 检测模型

前端的“检测模型”下拉框通过 `detector` 参数选择模型：

- `yolo`：默认优先读取 `model_results_export/yolo/best_yolo_mAP708.pt`，不存在时回退 `backend/models/invoice_yolo.pt`
- `faster_rcnn`：默认读取 `model_results_export/faster_rcnn/faster_rcnn_run/best.pt`
- `dfine_l`：默认读取 `model_results_export/dfine_l/best_dfine_l_mAP711.pth`

可用环境变量覆盖模型路径：

```bash
export YOLO_INVOICE_MODEL=/path/to/best_yolo.pt
export FASTER_RCNN_INVOICE_MODEL=/path/to/best_faster_rcnn.pt
export DFINE_L_INVOICE_MODEL=/path/to/best_dfine_l.pth
```

Faster R-CNN 推理需要额外安装 `torch` 和 `torchvision`。D-FINE-L 权重已接入状态展示和前端选择，Web 后端通过外部命令适配 D-FINE 推理。需要配置 `DFINE_PREDICT_COMMAND`：

```bash
export DFINE_PREDICT_COMMAND='python /path/to/predict_dfine_invoice.py --image {image} --weights {weights} --output {output} --conf {conf}'
```

项目启动脚本已经默认配置了本项目的适配命令：

```bash
scripts/start_backend_pyhon11_opencv.sh
```

适配脚本路径：

```text
scripts/predict_dfine_invoice.py
```

如果要真正运行 D-FINE-L，还需要把官方 D-FINE 仓库路径暴露给适配脚本：

```bash
export DFINE_REPO=/path/to/D-FINE
```

命令占位符含义：

- `{image}`：后端临时保存的输入图片路径
- `{weights}`：D-FINE-L 权重路径
- `{output}`：推理结果 JSON 输出路径
- `{conf}`：置信度阈值

JSON 输出格式：

```json
{
  "detections": [
    {
      "label": "invoice_no",
      "score": 0.95,
      "box": [100, 120, 240, 150]
    }
  ]
}
```

其中 `box` 默认为 `xyxy` 格式；如果输出为 `xywh`，需要额外提供 `"bbox_format": "xywh"`。

## 模块

- `main.py`：后端主入口
- `app.py`：兼容旧入口
- `invoice_app/__init__.py`：Flask app 工厂
- `invoice_app/routes.py`：API 路由层
- `invoice_app/preprocessing.py`：OpenCV 图像处理
- `invoice_app/ocr.py`：Tesseract OCR 备用引擎
- `invoice_app/paddle_ocr.py`：PaddleOCR 主识别引擎
- `invoice_app/parser.py`：发票字段解析
- `invoice_app/validators.py`：字段校验
- `invoice_app/pdf_utils.py`：PDF 文本提取和页面渲染
- `invoice_app/image_utils.py`：上传文件、图像编码和元数据工具
- `invoice_app/detectors.py`：YOLOv8n、Faster R-CNN、D-FINE-L 检测模型状态和推理入口
