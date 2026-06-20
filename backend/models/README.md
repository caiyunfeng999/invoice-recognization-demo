# YOLO 发票字段模型目录

将训练好的发票字段检测模型放在这里：

```text
backend/models/invoice_yolo.pt
```

也可以通过环境变量指定其他路径：

```bash
export YOLO_INVOICE_MODEL=/path/to/invoice_yolo.pt
```

建议 YOLO 类别名称使用以下英文或中文标签，后端会自动映射到结构化字段：

```text
invoice_type / 发票类型
invoice_code / 发票代码
invoice_no / invoice_number / 发票号码
invoice_date / date / 开票日期
checksum / check_code / 校验码
buyer_name / 购买方名称
buyer_tax_id / 购买方税号
seller_name / 销售方名称
seller_tax_id / 销售方税号
amount / 金额
tax / tax_amount / 税额
total / total_amount / 价税合计
drawer / issuer / 开票人
```
