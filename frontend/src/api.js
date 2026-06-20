import axios from "axios";


const API_BASE = "http://127.0.0.1:5001/api";


export async function processImage(file, method, params = {}) {
  const formData = new FormData();
  formData.append("file", file);
  Object.entries(params).forEach(([key, value]) => {
    formData.append(key, value);
  });
  const response = await axios.post(`${API_BASE}/process/${method}`, formData);
  return response.data;
}


export async function runOcr(file, options = {}) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("lang", options.lang || "chi_sim+eng");
  formData.append("psm", options.psm || 6);
  formData.append("engine", options.engine || "tesseract");
  formData.append("page", options.page || 1);
  formData.append("use_yolo", typeof options.use_yolo === "string" ? options.use_yolo : options.use_yolo ? "true" : "false");
  formData.append("yolo_confidence", options.yolo_confidence || 0.35);
  formData.append("detector", options.detector || "yolo");
  const response = await axios.post(`${API_BASE}/ocr`, formData);
  return response.data;
}


export async function detectYoloFields(file, options = {}) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("page", options.page || 1);
  formData.append("yolo_confidence", options.yolo_confidence || 0.35);
  formData.append("detector", options.detector || "yolo");
  const response = await axios.post(`${API_BASE}/yolo/detect`, formData);
  return response.data;
}


export async function getDetectorStatus() {
  const response = await axios.get(`${API_BASE}/detectors/status`);
  return response.data;
}
