import { useEffect, useMemo, useState } from "react";
import {
  uploadDataset,
  getDatasets,
  getDatasetPreview,
  getPreprocessors,
  getModels,
  getDefaultFeatures,
  runExperiment,
  getExperiments,
  fileUrl
} from "./api";

export default function App() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [preprocessors, setPreprocessors] = useState<any[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [defaultFeatures, setDefaultFeatures] = useState<string[]>([]);

  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedPreprocessor, setSelectedPreprocessor] = useState("ordinal_standard");
  const [selectedModel, setSelectedModel] = useState("agglomerative");

  const [datasetColumns, setDatasetColumns] = useState<string[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);

  const [nClusters, setNClusters] = useState(4);
  const [linkage, setLinkage] = useState("ward");
  const [metric, setMetric] = useState("euclidean");
  
  const [dbscanEps, setDbscanEps] = useState("auto");
  const [dbscanMinSamples, setDbscanMinSamples] = useState<string | number>("auto");
  
  const [experimentName, setExperimentName] = useState("Eksperimen Agglomerative Clustering");

  const [preview, setPreview] = useState<any>(null);
  const [activeResult, setActiveResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const recommendedFeaturesForCurrentDataset = useMemo(() => {
    if (datasetColumns.length === 0) return defaultFeatures;

    const availableDefaults = defaultFeatures.filter((col) => datasetColumns.includes(col));

    if (availableDefaults.length > 0) return availableDefaults;

    // Fallback universal: pilih kolom selain ID/label umum
    return datasetColumns.filter((col) => {
      const lower = col.toLowerCase();
      return !(
        lower === "id" ||
        lower.endsWith("id") ||
        lower.includes("name") ||
        lower.includes("email") ||
        lower.includes("label") ||
        lower.includes("target") ||
        lower.includes("class") ||
        lower.includes("engagementlevel") ||
        lower.includes("country")
      );
    });
  }, [datasetColumns, defaultFeatures]);

  async function refreshAll() {
    const [ds, pp, md, exp, features] = await Promise.all([
      getDatasets(),
      getPreprocessors(),
      getModels(),
      getExperiments(),
      getDefaultFeatures()
    ]);

    setDatasets(ds);
    setPreprocessors(pp);
    setModels(md);
    setExperiments(exp.reverse());
    setDefaultFeatures(features.default_gaming_features || []);

    if (!selectedDataset && ds.length > 0) {
      setSelectedDataset(ds[0].id);
    }
  }

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    if (selectedModel === "dbscan") {
      setExperimentName("Eksperimen DBSCAN Clustering");
    } else if (selectedModel === "kmeans") {
      setExperimentName("Eksperimen K-Means Clustering");
    } else {
      setExperimentName("Eksperimen Agglomerative Clustering");
    }
  }, [selectedModel]);

  useEffect(() => {
    if (selectedDataset) {
      handlePreview(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDataset]);

  async function handleUpload(e: any) {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    try {
      await uploadDataset(file);
      await refreshAll();
      alert("Dataset berhasil diupload.");
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handlePreview(autoSelectRecommended = false) {
    if (!selectedDataset) return;
    const data = await getDatasetPreview(selectedDataset);
    setPreview(data);

    const columns = data.dataset.column_names || [];
    setDatasetColumns(columns);

    const availableDefaults = (defaultFeatures || []).filter((col) => columns.includes(col));
    const nextRecommended = availableDefaults.length > 0
      ? availableDefaults
      : columns.filter((col: string) => {
          const lower = col.toLowerCase();
          return !(
            lower === "id" ||
            lower.endsWith("id") ||
            lower.includes("name") ||
            lower.includes("email") ||
            lower.includes("label") ||
            lower.includes("target") ||
            lower.includes("class") ||
            lower.includes("engagementlevel") ||
            lower.includes("country")
          );
        });

    if (autoSelectRecommended || selectedFeatures.length === 0) {
      setSelectedFeatures(nextRecommended);
    }
  }

  function toggleFeature(col: string) {
    setSelectedFeatures((prev) => {
      if (prev.includes(col)) {
        return prev.filter((item) => item !== col);
      }
      return [...prev, col];
    });
  }

  function selectRecommendedFeatures() {
    setSelectedFeatures(recommendedFeaturesForCurrentDataset);
  }

  function selectAllFeatures() {
    setSelectedFeatures(datasetColumns);
  }

  function clearFeatures() {
    setSelectedFeatures([]);
  }

  async function handleRunExperiment() {
    if (!selectedDataset) {
      alert("Pilih dataset terlebih dahulu.");
      return;
    }

    if (selectedFeatures.length < 2) {
      alert("Pilih minimal 2 kolom fitur untuk clustering.");
      return;
    }

    setLoading(true);
    try {
      const modelParams = selectedModel === "dbscan" 
        ? {
            eps: dbscanEps,
            min_samples: dbscanMinSamples === "auto" ? "auto" : Number(dbscanMinSamples)
          }
        : selectedModel === "kmeans"
        ? {
            n_clusters: Number(nClusters)
          }
        : {
            n_clusters: Number(nClusters),
            metric,
            linkage
          };

      const payload = {
        experiment_name: experimentName,
        dataset_id: selectedDataset,
        preprocessing_id: selectedPreprocessor,
        model_id: selectedModel,
        feature_columns: selectedFeatures,
        model_params: modelParams,
        dendrogram_sample_size: 100
      };

      const result = await runExperiment(payload);
      setActiveResult(result);
      await refreshAll();
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  }

  const result = activeResult || experiments[0];

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Machine Learning Clustering</p>
          <h1>Clustering Experiment Platform</h1>
          <p>
            Website untuk membandingkan dataset, fitur, teknik preprocessing, dan model clustering.
            Versi ini fokus pada Agglomerative Hierarchical Clustering, tetapi strukturnya siap ditambah metode lain.
          </p>
        </div>
      </header>

      <main className="grid">
        <section className="card">
          <h2>1. Upload & Simpan Dataset</h2>
          <p className="muted">Upload file CSV. Dataset akan disimpan di backend.</p>
          <input type="file" accept=".csv" onChange={handleUpload} />
          <div className="info-box">
            <strong>Dataset tersimpan:</strong> {datasets.length}
          </div>
        </section>

        <section className="card">
          <h2>2. Pilih Dataset</h2>
          <select value={selectedDataset} onChange={(e) => setSelectedDataset(e.target.value)}>
            <option value="">Pilih dataset</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.original_filename} ({d.rows} baris)
              </option>
            ))}
          </select>
          <button onClick={() => handlePreview(false)}>Preview Dataset</button>

          {preview && (
            <div className="preview">
              <h3>Preview</h3>
              <p>{preview.dataset.original_filename}</p>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      {preview.preview[0] &&
                        Object.keys(preview.preview[0]).map((col) => <th key={col}>{col}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.preview.map((row: any, idx: number) => (
                      <tr key={idx}>
                        {Object.values(row).map((val: any, i) => (
                          <td key={i}>{String(val)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>

        <section className="card wide-inner">
          <h2>3. Pilih Kolom Fitur</h2>
          <p className="muted">
            Centang kolom yang akan dipakai sebagai fitur clustering. Kolom yang tidak dicentang otomatis tidak masuk preprocessing.
          </p>

          <div className="feature-actions">
            <button onClick={selectRecommendedFeatures}>Gunakan Rekomendasi</button>
            <button onClick={selectAllFeatures}>Pilih Semua</button>
            <button onClick={clearFeatures}>Kosongkan</button>
          </div>

          <div className="info-box">
            <strong>Kolom terpilih:</strong> {selectedFeatures.length}
            <br />
            <span className="muted">Minimal pilih 2 kolom.</span>
          </div>

          <div className="checkbox-grid">
            {datasetColumns.length === 0 && (
              <p className="muted">Pilih dataset terlebih dahulu untuk melihat daftar kolom.</p>
            )}

            {datasetColumns.map((col) => {
              const checked = selectedFeatures.includes(col);
              const recommended = recommendedFeaturesForCurrentDataset.includes(col);

              return (
                <label key={col} className={`check-card ${checked ? "active" : ""}`}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleFeature(col)}
                  />
                  <span>{col}</span>
                  {recommended && <small>rekomendasi</small>}
                </label>
              );
            })}
          </div>
        </section>

        <section className="card">
          <h2>4. Teknik Preprocessing</h2>
          <select
            value={selectedPreprocessor}
            onChange={(e) => setSelectedPreprocessor(e.target.value)}
          >
            {preprocessors.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>

          <div className="info-box">
            <strong>Wajib ordinal encoding:</strong>
            <br />
            Easy = 0, Medium = 1, Hard = 2 karena Easy &lt; Medium &lt; Hard.
            <br />
            <br />
            Kolom kategori lain otomatis memakai One-Hot Encoding jika dipilih.
          </div>
        </section>

        <section className="card">
          <h2>5. Model Clustering</h2>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>

          <label>Nama eksperimen</label>
          <input
            value={experimentName}
            onChange={(e) => setExperimentName(e.target.value)}
          />

          {selectedModel === "dbscan" && (
            <>
              <label>Epsilon (eps)</label>
              <input
                type="text"
                value={dbscanEps}
                onChange={(e) => setDbscanEps(e.target.value)}
                placeholder="auto atau angka desimal (cth: 0.3)"
              />

              <label>Min Samples (minPts)</label>
              <input
                type="text"
                value={dbscanMinSamples}
                onChange={(e) => setDbscanMinSamples(e.target.value)}
                placeholder="auto atau angka (cth: 5)"
              />
            </>
          )}

          {selectedModel === "kmeans" && (
            <>
              <label>Jumlah Cluster</label>
              <input
                type="number"
                min={2}
                value={nClusters}
                onChange={(e) => setNClusters(Number(e.target.value))}
              />
            </>
          )}

          {selectedModel === "agglomerative" && (
            <>
              <label>Jumlah Cluster</label>
              <input
                type="number"
                min={2}
                value={nClusters}
                onChange={(e) => setNClusters(Number(e.target.value))}
              />

              <label>Linkage</label>
              <select value={linkage} onChange={(e) => setLinkage(e.target.value)}>
                <option value="ward">ward</option>
                <option value="complete">complete</option>
                <option value="average">average</option>
                <option value="single">single</option>
              </select>

              <label>Metric</label>
              <select value={metric} onChange={(e) => setMetric(e.target.value)}>
                <option value="euclidean">euclidean</option>
                <option value="manhattan">manhattan</option>
                <option value="cosine">cosine</option>
              </select>

              <p className="muted">
                Catatan: Jika linkage = ward, metric otomatis dianggap euclidean oleh backend.
              </p>
            </>
          )}

          <button className="primary" onClick={handleRunExperiment} disabled={loading}>
            {loading ? "Memproses..." : "Jalankan Eksperimen"}
          </button>
        </section>
      </main>

      <section className="card wide">
        <h2>6. Hasil Eksperimen</h2>
        {!result ? (
          <p className="muted">Belum ada hasil eksperimen.</p>
        ) : (
          <div>
            <h3>{result.experiment_name}</h3>
            <p className="muted">
              Dataset: {result.dataset?.filename} | Model: {result.model?.model_name} |
              Preprocessing: {result.preprocessing?.name}
            </p>

            <div className="info-box">
              <strong>Fitur input yang dipilih:</strong>
              <div className="mini-tags">
                {result.selected_input_features?.map((f: string) => (
                  <span key={f}>{f}</span>
                ))}
              </div>
              <strong>Fitur setelah preprocessing:</strong>
              <div className="mini-tags">
                {result.processed_output_features?.slice(0, 20).map((f: string) => (
                  <span key={f}>{f}</span>
                ))}
                {result.processed_output_features?.length > 20 && (
                  <span>+{result.processed_output_features.length - 20} fitur lain</span>
                )}
              </div>
            </div>

            <div className="metrics">
              <div>
                <span>Silhouette</span>
                <strong>{formatNumber(result.evaluation?.silhouette_score)}</strong>
              </div>
              <div>
                <span>Davies-Bouldin</span>
                <strong>{formatNumber(result.evaluation?.davies_bouldin_index)}</strong>
              </div>
              <div>
                <span>Calinski-Harabasz</span>
                <strong>{formatNumber(result.evaluation?.calinski_harabasz_score)}</strong>
              </div>
              <div>
                <span>Cluster</span>
                <strong>{result.evaluation?.n_clusters_detected}</strong>
              </div>
              {result.evaluation?.noise_count !== undefined && result.evaluation?.noise_count > 0 && (
                <div>
                  <span>Noise Data</span>
                  <strong>{result.evaluation.noise_count}</strong>
                </div>
              )}
            </div>

            <h3>Visualisasi</h3>
            <div className="image-grid">
              {result.images &&
                Object.entries(result.images).map(([key, url]: any) => (
                  <figure key={key}>
                    <img src={fileUrl(url)} alt={key} />
                    <figcaption>{key}</figcaption>
                  </figure>
                ))}
            </div>

            <h3>Download CSV</h3>
            <div className="download-list">
              {result.files &&
                Object.entries(result.files).map(([key, url]: any) => (
                  <a key={key} href={fileUrl(url)} target="_blank">
                    {key}
                  </a>
                ))}
            </div>

            <h3>Ringkasan Cluster</h3>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    {result.cluster_summary?.[0] &&
                      Object.keys(result.cluster_summary[0]).map((col: string) => (
                        <th key={col}>{col}</th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {result.cluster_summary?.map((row: any, idx: number) => (
                    <tr key={idx}>
                      {Object.values(row).map((val: any, i) => (
                        <td key={i}>{typeof val === "number" ? val.toFixed(3) : String(val)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <section className="card wide">
        <h2>7. Riwayat & Perbandingan Eksperimen</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Nama</th>
                <th>Dataset</th>
                <th>Preprocessing</th>
                <th>Model</th>
                <th>Fitur Input</th>
                <th>Cluster</th>
                <th>SI</th>
                <th>DBI</th>
                <th>CHI</th>
                <th>Aksi</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.id}>
                  <td>{e.experiment_name}</td>
                  <td>{e.dataset?.filename}</td>
                  <td>{e.preprocessing?.name}</td>
                  <td>{e.model?.model_name}</td>
                  <td>{e.selected_input_features?.length || 0}</td>
                  <td>{e.evaluation?.n_clusters_detected}</td>
                  <td>{formatNumber(e.evaluation?.silhouette_score)}</td>
                  <td>{formatNumber(e.evaluation?.davies_bouldin_index)}</td>
                  <td>{formatNumber(e.evaluation?.calinski_harabasz_score)}</td>
                  <td>
                    <button onClick={() => setActiveResult(e)}>Lihat</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function formatNumber(value: any) {
  if (value === null || value === undefined) return "-";
  if (typeof value !== "number") return value;
  return value.toFixed(4);
}
