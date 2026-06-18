const form = document.getElementById("analyze-form");
const resultEmpty = document.getElementById("result-empty");
const resultBody = document.getElementById("result-body");
const verdictBadge = document.getElementById("verdict-badge");
const metricWeight = document.getElementById("metric-weight");
const metricDuration = document.getElementById("metric-duration");
const metricDetected = document.getElementById("metric-detected");
const detectedList = document.getElementById("detected-list");
const missingList = document.getElementById("missing-list");
const warningList = document.getElementById("warning-list");
const submitButton = form.querySelector("button[type='submit']");

function clearList(listElement) {
  while (listElement.firstChild) {
    listElement.removeChild(listElement.firstChild);
  }
}

function appendItems(listElement, items, emptyMessage) {
  clearList(listElement);
  if (!items || items.length === 0) {
    const emptyItem = document.createElement("li");
    emptyItem.textContent = emptyMessage;
    listElement.appendChild(emptyItem);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    listElement.appendChild(li);
  });
}

function verdictClass(verdict) {
  if (verdict === "package can be shipped") return "ship";
  if (verdict === "normal") return "normal";
  return "blocked";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  submitButton.disabled = true;
  submitButton.textContent = "Analyzing...";

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Analysis failed.");

    resultEmpty.classList.add("hidden");
    resultBody.classList.remove("hidden");

    verdictBadge.textContent = data.verdict;
    verdictBadge.className = `verdict-badge ${verdictClass(data.verdict)}`;
    metricWeight.textContent = `${Number(data.package_weight_kg).toFixed(2)} kg`;
    metricDuration.textContent = `${Number(data.video_duration_seconds || 0).toFixed(2)} s`;
    metricDetected.textContent = `${(data.detected_labels || []).length}`;

    appendItems(detectedList, data.detected_labels || [], "No labels were detected.");
    appendItems(missingList, [...(data.blockers || []), ...(data.missing_declared_items || [])], "No missing or blocked items were found.");
    appendItems(warningList, data.warnings || [], "No warnings were raised.");
  } catch (error) {
    resultEmpty.classList.remove("hidden");
    resultBody.classList.add("hidden");
    resultEmpty.querySelector("p").textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Analyze package";
  }
});
