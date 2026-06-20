# 电子发票自动识别系统

React + Flask 课程项目，目标是实现“从发票图像或 PDF 到结构化数据”的自动处理系统。项目在老师 PPT 要求的模块化设计、主入口设计、按钮触发内部逻辑、处理结果返回、版本管理基础上，增加了 Web 化交互、PaddleOCR、PDF 文本直读、参数可调、处理记录、字段校验、JSON 导出和 YOLO 字段区域检测。

## 功能对应课程要求

| 课程要求 | 本项目实现 |
| --- | --- |
| 图形界面、图像加载、按钮触发、图像显示和结果输出 | React 前端完成图片/PDF 上传、处理按钮、预览窗口、结构化字段和 OCR 文本输出 |
| 图像去噪 | 中值滤波、高斯滤波，滤波核大小可调 |
| 灰度化与二值化 | 灰度化、Otsu 二值化、自适应阈值二值化 |
| 图像对比度增强 | CLAHE 自适应直方图均衡，增强参数可调 |
| 图像校正 | Canny 边缘、矩形轮廓检测、透视变换、旋转校正 |
| 文本区域定位 | 形态学膨胀 + 轮廓筛选 + 文本框标注 |
| OCR 文本识别 | PaddleOCR 为主，Tesseract 备用；支持 `chi_sim+eng` 和 PSM 参数 |
| 深度学习字段检测 | YOLO 检测发票关键字段区域，裁剪 ROI 后再 OCR |
| PDF 发票识别 | 电子 PDF 优先直接提取文字，扫描版 PDF 转图片后进入 OCR |
| 结构化信息提取 | 正则提取发票类型、代码、号码、日期、购买方、销售方、税号、金额、税额、开票人等 |
| 结果校验 | 对金额关系、税号、日期、字段缺失进行校验并提示需人工确认的字段 |
| 项目模块化 | 后端拆分为 `routes`、`preprocessing`、`ocr`、`paddle_ocr`、`parser`、`validators`、`pdf_utils`、`image_utils`；前端拆分为入口、API 层和主组件 |
| 版本管理 | 提供 `.gitignore`，可提交到 GitHub 仓库 |

## 项目结构

```text
invoice_ocr_web/
├── backend/
│   ├── main.py                     # Flask 主入口，保持简洁
│   ├── app.py                      # 兼容旧启动命令
│   ├── requirements.txt
│   └── invoice_app/
│       ├── __init__.py             # 创建 Flask app
│       ├── routes.py               # API 路由
│       ├── preprocessing.py        # 图像预处理、边缘检测、矫正、文本区域定位
│       ├── ocr.py                  # OCR 识别和置信度计算
│       ├── paddle_ocr.py           # PaddleOCR 识别
│       ├── yolo_detector.py        # YOLO 字段区域检测和字段裁剪 OCR
│       ├── parser.py               # 发票字段正则解析
│       ├── validators.py           # 字段合法性校验
│       ├── pdf_utils.py            # PDF 文本提取和页面渲染
│       └── image_utils.py          # 图片/PDF 读取、base64 编码、元数据
├── datasets/
│   └── invoice_yolo/               # YOLO 字段检测训练数据
├── docs/
│   ├── design_report.md            # 完整设计思路、问题解决、课堂代码对应
│   ├── core_code_explanation.md    # 核心代码讲解和答辩说明
│   └── yolo_upgrade.md             # YOLO 训练、部署和模型回传说明
└── frontend/
    ├── src/
    │   ├── main.jsx                # React 主入口
    │   ├── App.jsx                 # 页面和交互逻辑
    │   ├── api.js                  # 前后端接口封装
    │   └── styles.css
    └── package.json
```

## 后端

```bash
cd backend
conda activate pyhon11-opencv
pip install -r requirements.txt
python main.py
```

## 前端

```bash
cd frontend
npm install
npm run dev
```

前端地址：`http://127.0.0.1:5173`
后端地址：`http://127.0.0.1:5001`

## 使用流程

1. 点击“加载发票文件”上传发票照片、截图或 PDF。
2. 按需点击灰度化、去噪、二值化、对比度增强、边缘检测、透视矫正、文本区域定位等按钮。
3. 调整滤波核、Canny 阈值、OCR PSM 等参数以优化效果。
4. 点击“OCR 识别”，查看结构化字段、字段状态、OCR 原始文本、平均置信度和字段完整度。
5. 如需字段区域检测，点击“YOLO字段检测”或“YOLO+OCR识别”。
6. 点击“导出 JSON”保存结构化识别结果。

## 升级点

- 相比基础 tkinter 作业，本项目升级为 React + Flask 前后端分离架构。
- 支持图片普通发票、图片增值税专用发票、PDF 电子发票三类样例。
- 支持 PaddleOCR、Tesseract 和 PDF 文本直读三种识别路径。
- 每个处理操作都会返回处理说明、耗时、图像尺寸，并在前端形成处理流程记录。
- OCR 不只输出文本，还计算平均置信度和字段完整度。
- 识别结果增加字段校验提示，便于发现金额关系错误、税号异常和缺失字段。
- 支持 JSON 导出，便于后续接入数据库或财务报销系统。
- 加入 YOLO 深度学习字段检测模块，支持服务器训练模型后回传本地使用。

## YOLO 字段检测

默认模型路径：

```text
backend/models/invoice_yolo.pt
```

训练数据路径：

```text
datasets/invoice_yolo/
```

服务器训练完成后，只需要把 `best.pt` 回传并覆盖为：

```text
backend/models/invoice_yolo.pt
```

YOLO 训练和模型回传说明见 [docs/yolo_upgrade.md](docs/yolo_upgrade.md)。

## 测试样例

测试样例和字段级准确率说明见 [docs/test_cases.md](docs/test_cases.md)。

后端模块化设计与老师文档要求的对应关系见 [docs/backend_module_mapping.md](docs/backend_module_mapping.md)。

完整设计思路、问题解决和课堂代码复现说明见 [docs/design_report.md](docs/design_report.md)。

核心代码逐模块讲解和答辩话术见 [docs/core_code_explanation.md](docs/core_code_explanation.md)。

YOLO 字段检测升级、标注、服务器训练和模型回传说明见 [docs/yolo_upgrade.md](docs/yolo_upgrade.md)。

YOLOv10 半自动补样本、预标注和继续训练流程见 [docs/yolov10_semi_auto_training.md](docs/yolov10_semi_auto_training.md)。

## 当前限制

- OCR 无法保证所有票据 100% 正确，低清晰度、强反光、复杂版式仍可能需要人工修正。
- 当前 YOLO 数据集较小，金额、税额、销售方信息等小字段还需要更多标注样本继续提升。
- PDF 支持页码参数，课程样例一般为单页；多张发票批量合并仍可继续扩展。
- PaddleOCR 首次运行需要加载模型，第一次识别会比后续更慢。
