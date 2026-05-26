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
  const response = await axios.post(`${API_BASE}/ocr`, formData);
  return response.data;
}
