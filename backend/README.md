# Invoice OCR Backend

Run with the existing conda environment:

```bash
conda activate pyhon11-opencv
pip install -r requirements.txt
python app.py
```

The API runs on `http://127.0.0.1:5001`.

## API

- `GET /api/health`: 后端健康检查
- `GET /api/methods`: 获取可用图像处理方法
- `POST /api/process/<method>`: 执行图像处理并返回 base64 图像
- `POST /api/ocr`: 执行 Tesseract OCR，返回原始文本、结构化字段、平均置信度

## Modules

- `routes.py`: 负责接口输入输出
- `preprocessing.py`: 负责 OpenCV 图像处理
- `ocr.py`: 负责 pytesseract 调用和置信度统计
- `parser.py`: 负责发票字段正则解析
- `image_utils.py`: 负责图像读取、编码、元数据
