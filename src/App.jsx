import React, { useMemo, useState } from "react";
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
import { processImage, runOcr } from "./api";


const actions = [
  { key: "gray", label: "灰度化", icon: ImageIcon },
  { key: "denoise", label: "中值去噪", icon: Wand2 },
  { key: "gaussian", label: "高斯滤波", icon: Wand2 },
  { key: "otsu", label: "Otsu 二值化", icon: Braces },
  { key: "adaptive", label: "自适应二值化", icon: Braces },
  { key: "enhance", label: "对比度增强", icon: SlidersHorizontal },
  { key: "edges", label: "Canny 边缘", icon: Braces },
  { key: "sobel", label: "Sobel 边缘", icon: Braces },
  { key: "correct", label: "透视矫正", icon: RotateCcw },
  { key: "deskew", label: "旋转校正", icon: RotateCcw },
  { key: "regions", label: "文本区域定位", icon: ScanText },
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
  const [originalFile, setOriginalFile] = useState(null);
  const [originalPreview, setOriginalPreview] = useState("");
  const [ocrText, setOcrText] = useState("");
  const [fields, setFields] = useState({});
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const [params, setParams] = useState({
    kernel_size: 3,
    block_size: 35,
    c_value: 11,
    clip_limit: 2,
    low: 60,
    high: 180,
    psm: 6,
  });

  const canProcess = useMemo(() => Boolean(file), [file]);

  function updateParam(key, value) {
    setParams((current) => ({ ...current, [key]: value }));
  }

  function handleUpload(event) {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;

    const url = URL.createObjectURL(nextFile);
    setFile(nextFile);
    setOriginalFile(nextFile);
    setPreview(url);
    setOriginalPreview(url);
    setOcrText("");
    setFields({});
    setMetrics(null);
    setHistory([{ label: "加载图像", description: `${nextFile.name}，${Math.round(nextFile.size / 1024)} KB` }]);
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

  async function handleOcr() {
    if (!file) return;
    setLoading("ocr");
    setError("");

    try {
      const response = await runOcr(file, { psm: params.psm });
      setOcrText(response.text);
      setFields(response.fields);
      setMetrics({
        average_confidence: response.average_confidence,
        completion_score: response.completion_score,
        elapsed_ms: response.elapsed_ms,
        lang: response.lang,
        psm: response.psm,
      });
      setHistory((current) => [
        ...current,
        {
          label: "OCR 识别",
          description: `Tesseract ${response.lang}，PSM ${response.psm}，平均置信度 ${response.average_confidence}，字段完整度 ${Math.round(response.completion_score * 100)}%`,
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
    setHistory((current) => [...current, { label: "恢复原图", description: "已回到初始上传图像" }]);
    setError("");
  }

  function exportResult() {
    downloadJson("invoice-ocr-result.json", {
      fields,
      metrics,
      text: ocrText,
      history,
    });
  }

  return (
    <main className="app">
      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>电子发票自动识别系统</h1>
            <p>满足课程要求：预处理、边缘/轮廓、透视校正、形态学文本定位、OCR 与结构化输出</p>
          </div>
          <label className="upload">
            <FileImage size={18} />
            加载发票图像
            <input type="file" accept="image/*" onChange={handleUpload} />
          </label>
        </header>

        <div className="content">
          <section className="imagePanel">
            {preview ? <img src={preview} alt="发票预览" /> : <div className="empty">请选择一张发票图片</div>}
          </section>

          <aside className="sidePanel">
            <div className="toolGroup">
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
                OCR PSM
                <input type="number" min="3" max="13" value={params.psm} onChange={(e) => updateParam("psm", e.target.value)} />
              </label>
            </div>

            <div className="toolGroup">
              <div className="groupTitle">图像处理</div>
              <div className="buttons">
                {actions.map(({ key, label, icon: Icon }) => (
                  <button key={key} disabled={!canProcess || Boolean(loading)} onClick={() => handleProcess(key)} title={label}>
                    <Icon size={17} />
                    {loading === key ? "处理中..." : label}
                  </button>
                ))}
              </div>
            </div>

            <div className="toolGroup">
              <div className="groupTitle">识别与导出</div>
              <button className="primary" disabled={!canProcess || Boolean(loading)} onClick={handleOcr}>
                <ScanText size={18} />
                {loading === "ocr" ? "识别中..." : "OCR 识别"}
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

            {error && <div className="error">{error}</div>}
          </aside>
        </div>
      </section>

      <section className="results">
        <div className="resultBox">
          <h2>结构化字段</h2>
          {metrics && (
            <div className="metrics">
              <span>置信度 {metrics.average_confidence}</span>
              <span>完整度 {Math.round(metrics.completion_score * 100)}%</span>
              <span>耗时 {metrics.elapsed_ms} ms</span>
            </div>
          )}
          <dl>
            {Object.entries(fields).length ? (
              Object.entries(fields).map(([key, value]) => (
                <React.Fragment key={key}>
                  <dt>{key}</dt>
                  <dd>{value || "未识别"}</dd>
                </React.Fragment>
              ))
            ) : (
              <p className="muted">点击 OCR 识别后显示发票代码、号码、日期、金额、税号等信息。</p>
            )}
          </dl>
        </div>

        <div className="resultBox">
          <h2>处理流程记录</h2>
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

        <div className="resultBox textBox">
          <h2>OCR 原始文本</h2>
          <pre>{ocrText || "暂无识别文本"}</pre>
        </div>
      </section>
    </main>
  );
}
