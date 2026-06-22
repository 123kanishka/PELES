(function () {
  const API_BASE = window.PELES_API_BASE_URL;

  const form = document.getElementById("evaluate-form");
  const evaluateBtn = document.getElementById("evaluate-btn");
  const btnSpinner = evaluateBtn.querySelector(".btn-spinner");
  const btnLabel = evaluateBtn.querySelector(".btn-label");
  const errorText = document.getElementById("error-text");
  const backendBadge = document.getElementById("backend-badge");
  const result = document.getElementById("result");

  function setLoading(isLoading) {
    evaluateBtn.disabled = isLoading;
    btnSpinner.hidden = !isLoading;
    btnLabel.textContent = isLoading ? "Evaluating…" : "Evaluate";
  }

  function pct(p) {
    return `${Math.round(p * 100)}%`;
  }

  async function checkHealth() {
    try {
      const resp = await fetch(`${API_BASE}/api/health`);
      if (!resp.ok) throw new Error("unhealthy");
      const data = await resp.json();
      backendBadge.textContent = `backend: ${data.active_backend}`;
      backendBadge.className = "badge online";
    } catch {
      backendBadge.textContent = "backend offline";
      backendBadge.className = "badge offline";
    }
  }

  function renderResult(payload, modelAName, modelBName) {
    document.getElementById("result-winner").textContent = payload.winner_name;
    document.getElementById("prob-label-a").textContent = modelAName;
    document.getElementById("prob-label-b").textContent = modelBName;

    document.getElementById("prob-fill-a").style.width = pct(payload.prob_model_a);
    document.getElementById("prob-fill-b").style.width = pct(payload.prob_model_b);
    document.getElementById("prob-fill-tie").style.width = pct(payload.prob_tie);

    document.getElementById("prob-value-a").textContent = pct(payload.prob_model_a);
    document.getElementById("prob-value-b").textContent = pct(payload.prob_model_b);
    document.getElementById("prob-value-tie").textContent = pct(payload.prob_tie);

    document.getElementById("meta-backend").textContent = `model: ${payload.backend}`;
    document.getElementById("meta-latency").textContent = `${payload.latency_ms.toFixed(1)} ms`;

    result.hidden = false;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorText.textContent = "";

    const modelAName = document.getElementById("model-a-name").value.trim();
    const modelBName = document.getElementById("model-b-name").value.trim();
    const responseA = document.getElementById("response-a").value.trim();
    const responseB = document.getElementById("response-b").value.trim();
    const prompt = document.getElementById("prompt").value.trim();

    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/api/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_a_name: modelAName,
          model_b_name: modelBName,
          response_a: responseA,
          response_b: responseB,
          prompt: prompt,
        }),
      });

      if (!resp.ok) {
        const detail = await resp.json().catch(() => null);
        throw new Error(detail?.detail || `Request failed (${resp.status})`);
      }

      const payload = await resp.json();
      renderResult(payload, modelAName, modelBName);
    } catch (err) {
      errorText.textContent = err.message || "Something went wrong.";
      result.hidden = true;
    } finally {
      setLoading(false);
    }
  });

  checkHealth();
})();
