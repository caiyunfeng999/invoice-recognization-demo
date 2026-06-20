# 基于图像处理与 OCR 的电子发票自动识别系统

本项目是一个 React + Flask 前后端分离的电子发票自动识别系统，面向发票图片、截图和 PDF 文件，完成图像预处理、OCR 文本识别、关键字段检测、结构化解析、字段校验、结果导出和模型对比分析。

系统已接入三种深度学习字段检测模型：

- `YOLOv8n`：轻量、速度快，适合前端交互和半自动标注。
- `Faster R-CNN`：经典两阶段检测基线，召回率较高，但权重大、推理较慢。
- `D-FINE-L`：当前验证集 mAP50-95 最高，适合高精度离线对比。

## 主要功能

| 模块 | 实现内容 |
| --- | --- |
| 文件输入 | 支持图片和 PDF 发票上传 |
| 图像预处理 | 灰度化、去噪、二值化、CLAHE 增强、边缘检测、透视矫正、旋转校正 |
| 文本区域定位 | 形态学处理和轮廓筛选，返回文本框预览 |
| OCR 识别 | PaddleOCR 为主，Tesseract 备用；PDF 电子发票优先直接提取文本 |
| 字段检测 | 支持 YOLOv8n、Faster R-CNN、D-FINE-L 三模型选择 |
| 结构化解析 | 提取发票类型、代码、号码、日期、购买方、销售方、税号、金额、税额、价税合计、开票人等字段 |
| 字段校验 | 校验金额关系、税号格式、日期格式和缺失字段 |
| 前端交互 | 展示原图、处理图、检测框、OCR 文本、结构化字段和校验提示 |
| 结果导出 | 支持 JSON 结构化结果导出 |
| 论文材料 | 已生成模型对比、敏感性分析、鲁棒性分析和 ROC/FROC 风格图 |

## 项目结构

```text
invoice_ocr_web/
├── backend/
│   ├── main.py                         # Flask 后端入口
│   ├── requirements.txt
│   └── invoice_app/
│       ├── routes.py                   # API 路由
│       ├── preprocessing.py            # 图像预处理
│       ├── ocr.py                      # OCR 调度
│       ├── paddle_ocr.py               # PaddleOCR 封装
│       ├── yolo_detector.py            # YOLO 检测封装
│       ├── detectors.py                # 三模型统一检测接口
│       ├── parser.py                   # 发票字段解析
│       ├── validators.py               # 字段校验
│       ├── pdf_utils.py                # PDF 处理
│       └── image_utils.py              # 图片读取和编码
├── frontend/
│   ├── src/
│   │   ├── App.jsx                     # 前端主界面
│   │   ├── api.js                      # API 请求封装
│   │   └── styles.css
│   └── package.json
├── model_results_export/
│   ├── yolo/best_yolo_mAP708.pt
│   ├── faster_rcnn/faster_rcnn_run/best.pt
│   └── dfine_l/best_dfine_l_mAP711.pth
├── docs/
│   ├── thesis_draft_invoice_ocr_system.md
│   ├── invoice_detection_model_comparison.md
│   ├── nature_style_sensitivity_robustness_analysis.md
│   └── model_comparison_assets/
└── scripts/
    ├── start_backend_pyhon11_opencv.sh
    ├── setup_pyhon11_opencv_env.sh
    ├── predict_dfine_invoice.py
    └── generate_thesis_figures.py
```

## 环境准备

后端推荐使用已有虚拟环境：

```bash
conda activate pyhon11-opencv
cd /Users/caiyunfeng/invoice_ocr_web
bash scripts/setup_pyhon11_opencv_env.sh
```

前端依赖：

```bash
cd /Users/caiyunfeng/invoice_ocr_web/frontend
npm install
```

## 启动方式

启动后端：

```bash
cd /Users/caiyunfeng/invoice_ocr_web
bash scripts/start_backend_pyhon11_opencv.sh
```

启动前端：

```bash
cd /Users/caiyunfeng/invoice_ocr_web/frontend
npm run dev
```

默认地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:5001`

如果 `5173` 被占用，Vite 会自动切到下一个端口，以终端输出为准。

## 模型配置

后端启动脚本会默认设置以下模型路径：

```bash
YOLO_INVOICE_MODEL=/Users/caiyunfeng/invoice_ocr_web/model_results_export/yolo/best_yolo_mAP708.pt
FASTER_RCNN_INVOICE_MODEL=/Users/caiyunfeng/invoice_ocr_web/model_results_export/faster_rcnn/faster_rcnn_run/best.pt
DFINE_L_INVOICE_MODEL=/Users/caiyunfeng/invoice_ocr_web/model_results_export/dfine_l/best_dfine_l_mAP711.pth
```

D-FINE-L 依赖外部 D-FINE 运行环境，后端通过命令适配器调用：

```bash
DFINE_PREDICT_COMMAND="/opt/anaconda3/envs/pyhon11-opencv/bin/python /Users/caiyunfeng/invoice_ocr_web/scripts/predict_dfine_invoice.py --image {image} --weights {weights} --output {output} --conf {conf}"
```

前端会从 `/api/detectors/status` 获取可用模型，并在页面中提供模型选择。

## 模型结果

三种模型使用同一套 334 张人工复核发票数据和相同训练 / 验证划分进行对比。

| 模型 | mAP50-95 | mAP50 | Precision | Recall | F1 | 权重大小 | 适用场景 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D-FINE-L | 0.711 | 0.947 | 0.849 | 0.965 | 0.903 | 约 477MB | 高精度离线检测和论文对比 |
| YOLOv8n | 0.708 | 0.948 | 0.888 | 0.941 | 0.914 | 约 6.2MB | 前端实时交互、部署、半自动标注 |
| Faster R-CNN | 0.645 | 0.909 | 0.837 | 0.962 | 0.895 | 约 495MB | 两阶段检测基线和对照实验 |

当前结论：D-FINE-L 的 mAP50-95 最高；YOLOv8n 与 D-FINE-L 精度接近且模型最轻，更适合工程部署；Faster R-CNN 召回率较高，但整体 AP 和部署效率不占优。

## 使用流程

1. 打开前端页面，上传发票图片或 PDF。
2. 按需执行灰度化、去噪、二值化、增强、边缘检测、透视矫正等图像处理。
3. 在模型选择处选择 `YOLOv8n`、`Faster R-CNN` 或 `D-FINE-L`。
4. 点击 OCR 或字段检测相关按钮，查看检测框、OCR 文本和结构化字段。
5. 检查字段校验提示，必要时进行人工修正。
6. 导出 JSON 结果。

## 论文与图表

论文草稿：

- [docs/thesis_draft_invoice_ocr_system.md](docs/thesis_draft_invoice_ocr_system.md)

模型对比分析：

- [docs/invoice_detection_model_comparison.md](docs/invoice_detection_model_comparison.md)
- [docs/nature_style_sensitivity_robustness_analysis.md](docs/nature_style_sensitivity_robustness_analysis.md)

论文图表目录：

- [docs/model_comparison_assets/nature_analysis/](docs/model_comparison_assets/nature_analysis/)

重新生成论文图表：

```bash
cd /Users/caiyunfeng/invoice_ocr_web
/opt/anaconda3/envs/pyhon11-opencv/bin/python scripts/generate_thesis_figures.py
```

图表采用莫兰迪配色，标题和说明文字为中文，坐标轴保留英文；标注已调整为不遮挡图形主体。

## 常用脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/start_backend_pyhon11_opencv.sh` | 设置模型环境变量并启动 Flask 后端 |
| `scripts/setup_pyhon11_opencv_env.sh` | 安装 / 检查后端所需依赖 |
| `scripts/predict_dfine_invoice.py` | D-FINE-L 命令行推理适配器 |
| `scripts/generate_thesis_figures.py` | 生成论文模型对比图 |
| `scripts/run_yolo_param_sweep.py` | YOLO 参数组合搜索结果整理 |
| `scripts/auto_label_invoice_yolo.py` | 半自动标注辅助脚本 |
| `scripts/semiauto_label_with_yolo.py` | 使用 YOLO 对新图片预标注 |

## GitHub 上传说明

仓库中包含训练权重和实验压缩包，体积较大。上传 GitHub 时建议只提交代码、文档和小体积图表，不提交以下大文件：

- `model_results_export/**/*.pt`
- `model_results_export/**/*.pth`
- `*.tar.gz`
- 大规模训练数据和临时构建目录

如需复现实验，可在服务器重新训练或单独通过网盘 / Release 管理权重文件。

## 当前限制

- OCR 对低清晰度、强反光、严重倾斜或遮挡票据仍可能出错。
- D-FINE-L 在本地 Web 后端中需要额外配置 D-FINE 官方运行环境。
- 当前模型主要针对已有发票版式训练，跨版式泛化能力仍需要更多数据验证。
- 论文中的敏感性和 ROC/FROC 图属于基于现有实验结果的分析图，不等同于新增真实扰动实验。
