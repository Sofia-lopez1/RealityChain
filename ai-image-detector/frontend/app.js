(() => {
  const API = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || "";
  const ALLOWED = ["image/jpeg", "image/png", "image/webp"];
  let maxMb = 10;
  let currentFile = null;
  let previewUrl = null;

  const $ = (id) => document.getElementById(id);
  const el = {
    status: $("status"), notice: $("notice"), dropzone: $("dropzone"), file: $("file"),
    preview: $("preview"), previewImg: $("previewImg"), fileMeta: $("fileMeta"),
    analyze: $("analyze"), reset: $("reset"), error: $("error"), maxmb: $("maxmb"),
    result: $("result"), verdict: $("verdict"), verdictTitle: $("verdictTitle"), verdictSub: $("verdictSub"),
    thresholdTick: $("thresholdTick"), pAi: $("pAi"), certainty: $("certainty"),
    crops: $("crops"), time: $("time"), modelInfo: $("modelInfo"),
  };

  // Marcador de la probabilidad sobre la barra
  const marker = document.createElement("div");
  marker.className = "marker";
  marker.style.left = "0%";
  document.querySelector(".meter").appendChild(marker);

  /* ---------- Estado del servidor ---------- */
  async function checkHealth() {
    try {
      const res = await fetch(`${API}/api/health`);
      const data = await res.json();
      maxMb = data.max_upload_mb || maxMb;
      el.maxmb.textContent = maxMb;
      if (data.model_loaded) {
        setStatus("Modelo listo", "ok");
        el.notice.hidden = true;
      } else {
        setStatus("Modelo sin entrenar", "bad");
        showNotice("El servidor funciona, pero todavía no hay un modelo entrenado. Entrena uno siguiendo el README y reinicia el backend.");
      }
    } catch {
      setStatus("Sin conexión", "bad");
      showNotice("No se pudo conectar con el backend. ¿Está en marcha? Revisa API_BASE en frontend/config.js.");
    }
  }
  function setStatus(text, kind) {
    el.status.textContent = text;
    el.status.className = `badge badge-${kind}`;
  }
  function showNotice(text) {
    el.notice.textContent = text;
    el.notice.hidden = false;
  }

  /* ---------- Selección de archivo ---------- */
  function setFile(file) {
    showError("");
    if (!file) return;
    if (!ALLOWED.includes(file.type)) return showError("Formato no soportado. Usa JPG, PNG o WEBP.");
    if (file.size > maxMb * 1024 * 1024) return showError(`La imagen supera el máximo de ${maxMb} MB.`);

    currentFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    el.previewImg.src = previewUrl;
    el.fileMeta.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    el.dropzone.hidden = true;
    el.preview.hidden = false;
    el.result.hidden = true;
  }
  function reset() {
    currentFile = null;
    el.file.value = "";
    el.preview.hidden = true;
    el.dropzone.hidden = false;
    el.result.hidden = true;
    showError("");
  }
  function showError(msg) {
    el.error.textContent = msg;
    el.error.hidden = !msg;
  }

  el.file.addEventListener("change", () => setFile(el.file.files[0]));
  el.reset.addEventListener("click", reset);
  el.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.file.click(); }
  });
  ["dragenter", "dragover"].forEach((t) =>
    el.dropzone.addEventListener(t, (e) => { e.preventDefault(); el.dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) =>
    el.dropzone.addEventListener(t, (e) => { e.preventDefault(); el.dropzone.classList.remove("drag"); }));
  el.dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) setFile(item.getAsFile());
  });

  /* ---------- Análisis ---------- */
  el.analyze.addEventListener("click", async () => {
    if (!currentFile) return;
    showError("");
    el.analyze.disabled = true;
    el.analyze.textContent = "Analizando…";
    try {
      const body = new FormData();
      body.append("file", currentFile);
      const res = await fetch(`${API}/api/analyze`, { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
      renderResult(data);
    } catch (err) {
      showError(err.message || "No se pudo analizar la imagen.");
    } finally {
      el.analyze.disabled = false;
      el.analyze.textContent = "Analizar imagen";
    }
  });

  function renderResult(r) {
    const isAi = r.label === "ai";
    el.verdict.className = `verdict ${isAi ? "ai" : "real"}`;
    el.verdictTitle.textContent = isAi ? "Probablemente creada con IA" : "Probablemente real";
    el.verdictSub.textContent = isAi
      ? "El modelo detectó patrones típicos de imágenes generadas."
      : "El modelo no encontró patrones típicos de imágenes generadas.";

    marker.style.left = `${(r.probability_ai * 100).toFixed(1)}%`;
    el.thresholdTick.style.left = `${(r.threshold * 100).toFixed(1)}%`;
    el.pAi.textContent = `${(r.probability_ai * 100).toFixed(1)}%`;
    el.certainty.textContent = r.certainty.charAt(0).toUpperCase() + r.certainty.slice(1);
    el.crops.textContent = r.crops_analyzed;
    el.time.textContent = `${Math.round(r.inference_ms)} ms`;

    const m = r.model_metrics || {};
    const parts = [];
    if (m.accuracy != null) parts.push(`exactitud en validación ${(m.accuracy * 100).toFixed(1)}%`);
    if (m.auc != null) parts.push(`AUC ${m.auc.toFixed(3)}`);
    el.modelInfo.textContent = parts.length ? `Calidad del modelo: ${parts.join(" · ")}.` : "";
    if (r.crop_disagreement > 0.25) {
      el.modelInfo.textContent += " Los recortes de la imagen dieron resultados dispares: toma el resultado con cautela.";
    }

    el.result.hidden = false;
    el.result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  checkHealth();
})();
