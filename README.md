# 电子发票自动识别系统

React + Flask 课程项目，目标是实现“从发票图像到结构化数据”的自动处理系统。项目在老师 PPT 要求的模块化设计、主入口设计、按钮触发内部逻辑、处理结果返回、版本管理基础上，增加了 Web 化交互、参数可调、处理记录、OCR 置信度和 JSON 导出。

## 功能对应课程要求

| 课程要求 | 本项目实现 |
| --- | --- |
| 图形界面、图像加载、按钮触发、图像显示和结果输出 | React 前端完成图像上传、处理按钮、预览窗口、结构化字段和 OCR 文本输出 |
| 图像去噪 | 中值滤波、高斯滤波，滤波核大小可调 |
| 灰度化与二值化 | 灰度化、Otsu 二值化、自适应阈值二值化 |
| 图像对比度增强 | CLAHE 自适应直方图均衡，增强参数可调 |
| 图像校正 | Canny 边缘、矩形轮廓检测、透视变换、旋转校正 |
| 文本区域定位 | 形态学膨胀 + 轮廓筛选 + 文本框标注 |
| OCR 文本识别 | pytesseract，支持 `chi_sim+eng`，PSM 参数可调 |
| 结构化信息提取 | 正则提取发票类型、代码、号码、日期、购买方、销售方、税号、金额、税额、开票人等 |
| 项目模块化 | 后端拆分为 `routes`、`preprocessing`、`ocr`、`parser`、`image_utils`；前端拆分为入口、API 层和主组件 |
| 版本管理 | 提供 `.gitignore`，可提交到 GitHub 仓库 |

## 项目结构

```text
invoice_ocr_web/
├── backend/
│   ├── app.py                      # Flask 主入口，保持简洁
│   ├── requirements.txt
│   └── invoice_app/
│       ├── __init__.py             # 创建 Flask app
│       ├── routes.py               # API 路由
│       ├── preprocessing.py        # 图像预处理、边缘检测、矫正、文本区域定位
│       ├── ocr.py                  # OCR 识别和置信度计算
│       ├── parser.py               # 发票字段正则解析
│       └── image_utils.py          # 图像读取、base64 编码、元数据
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
python app.py
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

1. 点击“加载发票图像”上传发票照片或截图。
2. 按需点击灰度化、去噪、二值化、对比度增强、边缘检测、透视矫正、文本区域定位等按钮。
3. 调整滤波核、Canny 阈值、OCR PSM 等参数以优化效果。
4. 点击“OCR 识别”，查看结构化字段、OCR 原始文本、平均置信度和字段完整度。
5. 点击“导出 JSON”保存结构化识别结果。

## 升级点

- 相比基础 tkinter 作业，本项目升级为 React + Flask 前后端分离架构。
- 每个处理操作都会返回处理说明、耗时、图像尺寸，并在前端形成处理流程记录。
- OCR 不只输出文本，还计算平均置信度和字段完整度。
- 支持 JSON 导出，便于后续接入数据库或财务报销系统。
