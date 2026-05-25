const API_BASE = "http://127.0.0.1:8000";

export const fileUrl = (url: string) => {
  if (!url) return "";
  if (url.startsWith("http")) return url;
  return `${API_BASE}${url}`;
};

export async function uploadDataset(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/datasets/upload`, {
    method: "POST",
    body: formData
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Gagal upload dataset");
  }

  return res.json();
}

export async function getDatasets() {
  const res = await fetch(`${API_BASE}/api/datasets`);
  return res.json();
}

export async function getDatasetPreview(datasetId: string) {
  const res = await fetch(`${API_BASE}/api/datasets/${datasetId}/preview`);
  return res.json();
}

export async function getPreprocessors() {
  const res = await fetch(`${API_BASE}/api/registry/preprocessors`);
  return res.json();
}

export async function getModels() {
  const res = await fetch(`${API_BASE}/api/registry/models`);
  return res.json();
}

export async function getDefaultFeatures() {
  const res = await fetch(`${API_BASE}/api/registry/default-features`);
  return res.json();
}

export async function runExperiment(payload: any) {
  const res = await fetch(`${API_BASE}/api/experiments/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Gagal menjalankan eksperimen");
  }

  return res.json();
}

export async function getExperiments() {
  const res = await fetch(`${API_BASE}/api/experiments`);
  return res.json();
}
