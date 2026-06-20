import React, { useEffect, useMemo, useState } from "react";
import {
  Braces,
  Download,
  FileImage,
  Image as ImageIcon,
  RotateCcw,
  ScanText,
  SlidersHorizontal,
  Wand2,
} from "lucide-react";
import { detectYoloFields, getDetectorStatus, processImage, runOcr } from "./api";


const detectorFallbacks = [
  {
    key: "yolo",
    label: "YOLOv8n",
    available: true,
    metric: "mAP50-95 0.708 / mAP50 0.948",
    description: "轻量快速，适合默认识别。",
  },
  {
    key: "faster_rcnn",
    label: "Faster R-CNN",
    available: true,
    metric: "mAP50-95 0.645 / mAP50 0.909",
    description: "经典两阶段检测基线。",
  },
  {
    key: "dfine_l",
    label: "D-FINE-L",
    available: false,
    metric: "mAP50-95 0.711 / mAP50 0.947",
    description: "高精度离线检测模型。",
    note: "需要后端配置 DFINE_PREDICT_COMMAND 后才能直接推理。",
  },
];


const actionGroups = [
  {
    title: "灰度与增强",
    actions: [
      { key: "gray", label: "灰度化", icon: ImageIcon },
      { key: "invert", label: "灰度反转", icon: ImageIcon },
      { key: "gamma", label: "伽马变换", icon: SlidersHorizontal },
      { key: "stretch", label: "线性拉伸", icon: SlidersHorizontal },
      { key: "hist_equalize", label: "直方图均衡", icon: SlidersHorizontal },
      { key: "enhance", label: "对比度增强", icon: SlidersHorizontal },
    ],
  },
  {
    title: "滤波去噪",
    actions: [
      { key: "mean", label: "均值滤波", icon: Wand2 },
      { key: "denoise", label: "中值去噪", icon: Wand2 },
      { key: "gaussian", label: "高斯滤波", icon: Wand2 },
    ],
  },
  {
    title: "阈值分割",
    actions: [
      { key: "otsu", label: "Otsu 二值化", icon: Braces },
      { key: "iter_threshold", label: "迭代阈值", icon: Braces },
      { key: "adaptive", label: "自适应二值化", icon: Braces },
    ],
  },
  {
    title: "边缘与锐化",
    actions: [
      { key: "edges", label: "Canny 边缘", icon: Braces },
      { key: "sobel", label: "Sobel 边缘", icon: Braces },
      { key: "laplacian", label: "拉普拉斯锐化", icon: SlidersHorizontal },
    ],
  },
  {
    title: "形态学处理",
    actions: [
      { key: "morph_open", label: "开运算", icon: Braces },
      { key: "morph_close", label: "闭运算", icon: Braces },
      { key: "morph_gradient", label: "形态学梯度", icon: Braces },
    ],
  },
  {
    title: "校正与定位",
    actions: [
      { key: "correct", label: "透视矫正", icon: RotateCcw },
      { key: "deskew", label: "旋转校正", icon: RotateCcw },
      { key: "regions", label: "文本区域定位", icon: ScanText },
    ],
  },
];


function dataUrlToFile(dataUrl, filename) {
  const [header, base64] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)?.[1] || "image/png";
  const binary = atob(base64);
  const array = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    array[i] = binary.charCodeAt(i);
  }
  return new File([array], filename, { type: mime });
}


function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}


export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [previewType, setPreviewType] = useState("");
  const [originalFile, setOriginalFile] = useState(null);
  const [originalPreview, setOriginalPreview] = useState("");
  const [originalPreviewType, setOriginalPreviewType] = useState("");
  const [ocrText, setOcrText] = useState("");
  const [fields, setFields] = useState({});
  const [fieldChecks, setFieldChecks] = useState({});
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const [resultTab, setResultTab] = useState("fields");
  const [detectors, setDetectors] = useState(detectorFallbacks);
  const [params, setParams] = useState({
    kernel_size: 3,
    block_size: 35,
    c_value: 11,
    clip_limit: 2,
    low: 60,
    high: 180,
    psm: 6,
    engine: "paddle",
    page: 1,
    yolo_confidence: 0.35,
    detector: "yolo",
  });

  const canProcess = useMemo(() => Boolean(file), [file]);
  const recognitionLabels = {
    ocr: "OCR 识别中",
    "auto-ocr": "智能识别中",
    "detector-detect": "字段检测中",
    "detector-ocr": "检测模型+OCR 识别中",
  };
  const recognitionLabel = recognitionLabels[loading] || "";
  const selectedDetector = useMemo(
    () => detectors.find((item) => item.key === params.detector) || detectorFallbacks[0],
    [detectors, params.detector],
  );

  useEffect(() => {
    let ignore = false;
    getDetectorStatus()
      .then((data) => {
        if (ignore) return;
        setDetectors(data.detectors || []);
        if (data.default && !params.detector) {
          updateParam("detector", data.default);
        }
      })
      .catch(() => {
        if (!ignore) {
          setDetectors(detectorFallbacks);
        }
      });
    return () => {
      ignore = true;
    };
  }, []);

  function updateParam(key, value) {
    setParams((current) => ({ ...current, [key]: value }));
  }

  function updateField(key, value) {
    setFields((current) => ({ ...current, [key]: value }));
  }

  function handleUpload(event) {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;

    const url = URL.createObjectURL(nextFile);
    const type = nextFile.type === "application/pdf" || nextFile.name.toLowerCase().endsWith(".pdf") ? "pdf" : "image";
    setFile(nextFile);
    setOriginalFile(nextFile);
    setPreview(url);
    setPreviewType(type);
    setOriginalPreview(url);
    setOriginalPreviewType(type);
    setOcrText("");
    setFields({});
    setFieldChecks({});
    setMetrics(null);
    setHistory([{ label: "加载文件", description: `${nextFile.name}，${Math.round(nextFile.size / 1024)} KB` }]);
    updateParam("page", 1);
    setError("");
  }

  async function handleProcess(method) {
    if (!file) return;
    setLoading(method);
    setError("");

    try {
      const response = await processImage(file, method, params);
      const imageUrl = `data:image/png;base64,${response.image}`;
      setPreview(imageUrl);
      setPreviewType("image");
      setFile(dataUrlToFile(imageUrl, `processed-${method}.png`));
      setHistory((current) => [
        ...current,
        {
          label: response.label,
          description: `${response.description}；耗时 ${response.elapsed_ms} ms；尺寸 ${response.metadata.width}x${response.metadata.height}`,
        },
      ]);
    } catch (err) {
      setError(err.response?.data?.error || "图像处理失败");
    } finally {
      setLoading("");
    }
  }

  async function handleYoloDetect() {
    if (!file) return;
    setLoading("detector-detect");
    setError("");

    try {
      const response = await detectYoloFields(file, {
        page: params.page,
        yolo_confidence: params.yolo_confidence,
        detector: params.detector,
      });
      const imageUrl = `data:image/png;base64,${response.image}`;
      setPreview(imageUrl);
      setPreviewType("image");
      setFile(dataUrlToFile(imageUrl, `${params.detector}-detected-fields.png`));
      setHistory((current) => [
        ...current,
        {
          label: `${response.detector?.label || selectedDetector.label}字段检测`,
          description: `检测到 ${response.detection_count} 个字段区域；耗时 ${response.elapsed_ms} ms；尺寸 ${response.metadata.width}x${response.metadata.height}`,
        },
      ]);
    } catch (err) {
      setError(err.response?.data?.error || `${selectedDetector.label} 字段检测失败`);
    } finally {
      setLoading("");
    }
  }

  async function handleOcr(yoloMode = "false") {
    if (!file) return;
    const mode = yoloMode === true ? "true" : yoloMode || "false";
    const loadingKey = mode === "auto" ? "auto-ocr" : mode === "true" ? "detector-ocr" : "ocr";
    const actionLabel = mode === "auto" ? "智能识别" : mode === "true" ? `${selectedDetector.label}+OCR识别` : "OCR 识别";
    setLoading(loadingKey);
    setError("");

    try {
      const response = await runOcr(file, {
        psm: params.psm,
        engine: params.engine,
        page: params.page,
        use_yolo: mode,
        yolo_confidence: params.yolo_confidence,
        detector: params.detector,
      });
      setOcrText(response.text);
      setFields(response.fields);
      setFieldChecks(response.field_checks || {});
      setResultTab("fields");
      if (response.image) {
        setPreview(`data:image/png;base64,${response.image}`);
        setPreviewType("image");
      }
      setMetrics({
        average_confidence: response.average_confidence,
        completion_score: response.completion_score,
        elapsed_ms: response.elapsed_ms,
        lang: response.lang,
        psm: response.psm,
        engine: response.engine,
        metadata: response.metadata,
        yolo_decision: response.yolo_decision,
        detector: response.detector,
      });
      setHistory((current) => [
        ...current,
        {
          label: actionLabel,
          description: `${response.engine === "pdf_text" ? "PDF 文本直读" : response.engine?.includes("+paddle") ? `${response.detector?.label || selectedDetector.label}字段检测 + PaddleOCR，检测 ${response.detection_count || 0} 个区域` : response.engine === "paddle" ? "PaddleOCR" : `Tesseract ${response.lang}，PSM ${response.psm}`}，平均置信度 ${response.average_confidence}，字段完整度 ${Math.round(response.completion_score * 100)}%${response.metadata?.file_type === "pdf" ? `，第 ${response.metadata.source_page}/${response.metadata.pages} 页` : ""}${response.yolo_decision ? `，${response.yolo_decision}` : ""}`,
        },
      ]);
    } catch (err) {
      setError(err.response?.data?.error || "OCR 识别失败");
    } finally {
      setLoading("");
    }
  }

  function resetImage() {
    if (!originalFile || !originalPreview) return;
    setFile(originalFile);
    setPreview(originalPreview);
    setPreviewType(originalPreviewType);
    setHistory((current) => [...current, { label: "恢复原图", description: "已回到初始上传图像" }]);
    setError("");
  }

  function exportResult() {
    downloadJson("invoice-ocr-result.json", {
      fields,
      fieldChecks,
      metrics,
      text: ocrText,
      history,
    });
  }

  return (
    <main className="app">
      <section className="workspace">
        <header className="topbar">
          <div className="titleBlock">
            <div className="titleLine">
              <h1>电子发票自动识别系统</h1>
            </div>
          </div>
          {recognitionLabel && (
            <div className="recognitionProgress" role="status" aria-live="polite">
              <div className="recognitionProgressText">{recognitionLabel}</div>
              <div className="progressTrack">
                <div className="progressBar" />
              </div>
            </div>
          )}
          <label className="upload">
            <FileImage size={18} />
            加载发票文件
            <input type="file" accept="image/*,.pdf,application/pdf" onChange={handleUpload} />
          </label>
        </header>

        <div className="content">
          <section className="imagePanel">
            {preview && previewType === "pdf" ? (
              <iframe src={preview} title="PDF 发票预览" />
            ) : preview ? (
              <img src={preview} alt="发票预览" />
            ) : (
              <div className="empty">请选择一张发票图片或 PDF</div>
            )}
          </section>

          <aside className="sidePanel">
            <div className={`panelSection paramsSection ${previewType === "pdf" ? "withPdfPage" : ""}`}>
              <div className="groupTitle">可调参数</div>
              <label className="field">
                滤波核
                <input type="number" min="3" step="2" value={params.kernel_size} onChange={(e) => updateParam("kernel_size", e.target.value)} />
              </label>
              <label className="field">
                自适应块
                <input type="number" min="3" step="2" value={params.block_size} onChange={(e) => updateParam("block_size", e.target.value)} />
              </label>
              <label className="field">
                Canny 阈值
                <span className="pair">
                  <input type="number" value={params.low} onChange={(e) => updateParam("low", e.target.value)} />
                  <input type="number" value={params.high} onChange={(e) => updateParam("high", e.target.value)} />
                </span>
              </label>
              <label className="field">
                OCR 引擎
                <select value={params.engine} onChange={(e) => updateParam("engine", e.target.value)}>
                  <option value="paddle">PaddleOCR</option>
                  <option value="tesseract">Tesseract</option>
                </select>
              </label>
              <label className="field">
                OCR PSM
                <input type="number" min="3" max="13" value={params.psm} onChange={(e) => updateParam("psm", e.target.value)} />
              </label>
              <label className="field">
                检测模型
                <select value={params.detector} onChange={(e) => updateParam("detector", e.target.value)}>
                  {detectors.map((item) => (
                    <option key={item.key} value={item.key}>
                      {item.label}{item.available ? "" : "（未启用）"}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                检测阈值
                <input type="number" min="0.1" max="0.9" step="0.05" value={params.yolo_confidence} onChange={(e) => updateParam("yolo_confidence", e.target.value)} />
              </label>
              {previewType === "pdf" && (
                <label className="field">
                  PDF 页码
                  <input type="number" min="1" value={params.page} onChange={(e) => updateParam("page", e.target.value)} />
                </label>
              )}
            </div>

            <div className="panelSection processSection">
              <div className="groupTitle">图像处理</div>
              <button className="optimizeButton" disabled={!canProcess || Boolean(loading)} onClick={() => handleProcess("auto_optimize")}>
                <Wand2 size={18} />
                {loading === "auto_optimize" ? "优化中..." : "一键优化发票图片"}
              </button>
              <div className="buttonCategories">
                {actionGroups.map((group) => (
                  <div className="buttonCategory" key={group.title}>
                    <div className="categoryTitle">{group.title}</div>
                    <div className="buttons">
                      {group.actions.map(({ key, label, icon: Icon }) => (
                        <button key={key} disabled={!canProcess || Boolean(loading)} onClick={() => handleProcess(key)} title={label}>
                          <Icon size={17} />
                          {loading === key ? "处理中..." : label}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="panelSection actionSection">
              <div className="groupTitle">识别与导出</div>
              <div className="actionGrid">
                <button className="primary" disabled={!canProcess || Boolean(loading)} onClick={() => handleOcr(false)}>
                  <ScanText size={18} />
                  {loading === "ocr" ? "识别中..." : "OCR 识别"}
                </button>
                <button disabled={!canProcess || Boolean(loading)} onClick={() => handleOcr("auto")}>
                  <ScanText size={17} />
                  {loading === "auto-ocr" ? "识别中..." : "智能识别"}
                </button>
                <button disabled={!canProcess || Boolean(loading) || selectedDetector.available === false} onClick={handleYoloDetect}>
                  <ScanText size={17} />
                  {loading === "detector-detect" ? "检测中..." : `${selectedDetector.label}检测`}
                </button>
                <button disabled={!canProcess || Boolean(loading) || selectedDetector.available === false} onClick={() => handleOcr(true)}>
                  <ScanText size={17} />
                  {loading === "detector-ocr" ? "识别中..." : `${selectedDetector.label}+OCR`}
                </button>
                <button disabled={!originalPreview || Boolean(loading)} onClick={resetImage}>
                  <RotateCcw size={17} />
                  恢复原图
                </button>
                <button disabled={!Object.keys(fields).length} onClick={exportResult}>
                  <Download size={17} />
                  导出 JSON
                </button>
              </div>
            </div>

            {error && <div className="error">{error}</div>}
            {selectedDetector.available === false && (
              <div className="detectorNote">
                {selectedDetector.label} 当前不可直接推理：{selectedDetector.note || "模型权重或运行时未配置"}
              </div>
            )}
          </aside>
        </div>
      </section>

      <section className="results">
        <div className="resultsShell">
          <div className="resultsTop">
            <div>
              <h2>{resultTab === "fields" ? "结构化字段" : resultTab === "history" ? "处理流程记录" : "OCR 原始文本"}</h2>
              <p>
                {resultTab === "fields"
                  ? Object.keys(fields).length ? `${Object.keys(fields).length} 项字段` : "等待识别结果"
                  : resultTab === "history"
                    ? `${history.length} 步处理`
                    : ocrText ? `${ocrText.length} 字符` : "暂无识别文本"}
              </p>
            </div>
            <div className="resultTabs">
              <button className={resultTab === "fields" ? "active" : ""} onClick={() => setResultTab("fields")}>字段</button>
              <button className={resultTab === "history" ? "active" : ""} onClick={() => setResultTab("history")}>流程</button>
              <button className={resultTab === "text" ? "active" : ""} onClick={() => setResultTab("text")}>文本</button>
            </div>
          </div>

          <div className="resultContent">
            {resultTab === "fields" && (
              <div className="resultPane">
                {metrics && (
                  <div className="metrics">
                    <span>置信度 {metrics.average_confidence}</span>
                    <span>完整度 {Math.round(metrics.completion_score * 100)}%</span>
                    <span>耗时 {metrics.elapsed_ms} ms</span>
                    <span>{metrics.engine === "pdf_text" ? "PDF 文本" : metrics.engine?.includes("+paddle") ? `${metrics.detector?.label || selectedDetector.label}+PaddleOCR` : metrics.engine === "paddle" ? "PaddleOCR" : "Tesseract"}</span>
                    {metrics.metadata?.file_type === "pdf" && <span>第 {metrics.metadata.source_page}/{metrics.metadata.pages} 页</span>}
                    {metrics.yolo_decision && <span>{metrics.yolo_decision}</span>}
                  </div>
                )}
                <dl>
                  {Object.entries(fields).length ? (
                    Object.entries(fields).map(([key, value]) => (
                      <React.Fragment key={key}>
                        <dt>{key}</dt>
                        <dd>
                          <input
                            className="fieldEdit"
                            value={value || ""}
                            placeholder="未识别"
                            onChange={(event) => updateField(key, event.target.value)}
                          />
                          {fieldChecks[key] && (
                            <span className={`fieldStatus ${fieldChecks[key].level}`}>
                              {fieldChecks[key].message}
                            </span>
                          )}
                        </dd>
                      </React.Fragment>
                    ))
                  ) : (
                    <p className="muted">点击 OCR 识别后显示发票代码、号码、日期、金额、税号等信息。</p>
                  )}
                </dl>
              </div>
            )}

            {resultTab === "history" && (
              <div className="resultPane">
                <ol className="history">
                  {history.length ? (
                    history.map((item, index) => (
                      <li key={`${item.label}-${index}`}>
                        <strong>{item.label}</strong>
                        <span>{item.description}</span>
                      </li>
                    ))
                  ) : (
                    <li className="muted">暂无处理记录</li>
                  )}
                </ol>
              </div>
            )}

            {resultTab === "text" && (
              <div className="resultPane">
                <pre>{ocrText || "暂无识别文本"}</pre>
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
