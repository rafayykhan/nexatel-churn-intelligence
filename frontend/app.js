/* NexaTel Retention Intelligence — client logic.
 *
 * The API base resolves in this order:
 *   1. ?api=<url> in the query string   (quick testing against any backend)
 *   2. window.NEXATEL_API                (set by config.js when deployed split)
 *   3. same origin                       (FastAPI serving this folder — default)
 * That last case is how the project deploys on Render: one service, no CORS.
 */
const API = (() => {
  const q = new URLSearchParams(location.search).get("api");
  if (q) return q.replace(/\/$/, "");
  if (window.NEXATEL_API) return window.NEXATEL_API.replace(/\/$/, "");
  return "";
})();

const $ = (id) => document.getElementById(id);
const ADDONS = [
  ["online_security", "Online security"],
  ["online_backup", "Online backup"],
  ["device_protection", "Device protection"],
  ["tech_support", "Tech support"],
  ["streaming_tv", "Streaming TV"],
  ["streaming_movies", "Streaming movies"],
];

const ARC_LEN = 283;           // length of the 180° gauge path
const money = (n) =>
  "$" + Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 });

let META = null;

/* ------------------------------------------------------------------ */
/* add-on checkboxes                                                   */
/* ------------------------------------------------------------------ */
function buildAddons() {
  $("addons").innerHTML = ADDONS.map(
    ([name, label]) => `
    <label class="addon" data-for="${name}">
      <input type="checkbox" name="${name}" /> <span>${label}</span>
    </label>`
  ).join("");

  $("addons").addEventListener("change", (e) => {
    e.target.closest(".addon")?.classList.toggle("on", e.target.checked);
  });
}

/* Add-ons only exist for internet subscribers. Rather than let an agent
 * submit "tech support: Yes, internet: No" — a combination that appears
 * nowhere in the training data — the checkboxes lock when internet is No. */
function syncInternetLock() {
  const off = document.querySelector('[name="internet_service"]').value === "No";
  document.querySelectorAll(".addon").forEach((el) => {
    const box = el.querySelector("input");
    if (off) { box.checked = false; el.classList.remove("on"); }
    box.disabled = off;
    el.classList.toggle("disabled", off);
  });
}

function readForm() {
  const f = $("churn-form");
  const data = Object.fromEntries(new FormData(f).entries());
  const noNet = data.internet_service === "No";

  ADDONS.forEach(([name]) => {
    const box = f.querySelector(`[name="${name}"]`);
    data[name] = noNet ? "No internet service" : box.checked ? "Yes" : "No";
  });

  if (data.phone_service === "No") data.multiple_lines = "No phone service";

  data.tenure = parseInt(data.tenure || "0", 10);
  data.senior_citizen = parseInt(data.senior_citizen, 10);
  data.monthly_charges = parseFloat(data.monthly_charges || "0");
  data.total_charges =
    data.total_charges === "" ? null : parseFloat(data.total_charges);
  return data;
}

function fillForm(c) {
  const f = $("churn-form");
  Object.entries(c).forEach(([k, v]) => {
    const el = f.querySelector(`[name="${k}"]`);
    if (!el) return;
    if (el.type === "checkbox") {
      el.checked = v === "Yes";
      el.closest(".addon").classList.toggle("on", el.checked);
    } else {
      el.value = v;
    }
  });
  syncInternetLock();
}

/* ------------------------------------------------------------------ */
/* rendering a score                                                   */
/* ------------------------------------------------------------------ */
function renderResult(r) {
  $("result-empty").hidden = true;
  $("result").hidden = false;

  const pct = r.risk_score;
  $("score").textContent = pct;

  const chip = $("risk-chip");
  chip.textContent = r.risk_level + " risk";
  chip.className = "chip " + r.risk_level.toLowerCase();

  // Arc fills proportionally; the tick shows where the company's outreach
  // line sits, so an agent can see how far past it this customer is.
  requestAnimationFrame(() => {
    $("arc").style.strokeDashoffset = ARC_LEN * (1 - pct / 100);
  });

  const t = r.decision_threshold;
  const angle = Math.PI * (1 - t);
  const cx = 110, cy = 118, r1 = 84, r2 = 96;
  const tick = $("threshold-tick");
  tick.setAttribute("x1", cx + r1 * Math.cos(angle));
  tick.setAttribute("y1", cy - r1 * Math.sin(angle));
  tick.setAttribute("x2", cx + r2 * Math.cos(angle));
  tick.setAttribute("y2", cy - r2 * Math.sin(angle));
  $("tick-note").textContent = `outreach line at ${Math.round(t * 100)}%`;

  $("verdict").innerHTML = r.flagged_for_outreach
    ? `Above the outreach line — <strong>contact this customer</strong>.`
    : `Below the outreach line. <strong>No offer needed today.</strong>`;

  const bar = (f, max) => `
    <li>
      <div class="f-top"><span class="f-label">${f.label}</span>
      <span class="f-impact">${f.impact > 0 ? "+" : ""}${f.impact.toFixed(3)}</span></div>
      <div class="f-bar"><div class="f-fill" style="width:${Math.min(
        100, (Math.abs(f.impact) / max) * 100
      )}%"></div></div>
    </li>`;

  const all = [...r.risk_factors, ...r.protective_factors];
  const max = Math.max(...all.map((f) => Math.abs(f.impact)), 0.01);

  $("factor-count").textContent = `top ${r.risk_factors.length}`;
  $("risk-factors").innerHTML =
    r.risk_factors.map((f) => bar(f, max)).join("") ||
    `<li class="f-label">Nothing is pushing this customer toward cancelling.</li>`;

  $("protective-block").hidden = r.protective_factors.length === 0;
  $("protective-factors").innerHTML = r.protective_factors
    .map((f) => bar(f, max))
    .join("");

  $("action-text").textContent = r.recommended_action;
  $("revenue").textContent = money(r.revenue_at_risk_annual);
  $("model-note").textContent = `${r.model_name} · threshold ${t} · SHAP explanations`;

  $("result-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(msg) {
  const old = document.querySelector(".err");
  if (old) old.remove();
  const el = document.createElement("div");
  el.className = "err";
  el.textContent = msg;
  $("churn-form").appendChild(el);
}

async function score(e) {
  e.preventDefault();
  const btn = $("submit-btn");
  document.querySelector(".err")?.remove();
  btn.disabled = true;
  btn.textContent = "Scoring…";
  try {
    const res = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Server returned ${res.status}`);
    }
    renderResult(await res.json());
  } catch (err) {
    showError(
      `Could not score this customer: ${err.message}. Check the API is running, then try again.`
    );
  } finally {
    btn.disabled = false;
    btn.textContent = "Score customer";
  }
}

/* ------------------------------------------------------------------ */
/* samples + status                                                    */
/* ------------------------------------------------------------------ */
const SWATCH = { High: "var(--rose)", Medium: "var(--amber)", Low: "var(--green)" };

async function loadSamples() {
  try {
    const { samples } = await (await fetch(`${API}/api/samples`)).json();
    $("samples").innerHTML = samples
      .map((s, i) => {
        const level = s.label.split(" ")[0];
        const short = s.label.split("—")[1]?.trim() || s.label;
        return `<button type="button" class="sample-btn" data-i="${i}">
          <span class="swatch" style="background:${SWATCH[level] || "var(--ink-faint)"}"></span>${short}
        </button>`;
      })
      .join("");
    $("samples").addEventListener("click", (e) => {
      const b = e.target.closest(".sample-btn");
      if (b) fillForm(samples[+b.dataset.i].customer);
    });
  } catch {
    $("samples").innerHTML = "";
  }
}

async function loadHealth() {
  try {
    const h = await (await fetch(`${API}/health`)).json();
    META = h;
    $("status-dot").className = "dot ok";
    $("status-text").textContent = `${h.model} · ready`;
  } catch {
    $("status-dot").className = "dot bad";
    $("status-text").textContent = "API unreachable";
  }
}

/* ------------------------------------------------------------------ */
/* insights tab                                                        */
/* ------------------------------------------------------------------ */
let insightsLoaded = false;

const FIGURES = [
  ["02_churn_by_category.png", "Churn rate by account attribute"],
  ["05_segment_heatmap.png", "Contract × tenure — where churn concentrates"],
  ["07_roc_curves.png", "ROC curves, all five candidates"],
  ["10_shap_summary.png", "SHAP summary — direction and magnitude"],
];

async function loadInsights() {
  if (insightsLoaded) return;
  insightsLoaded = true;
  try {
    const d = await (await fetch(`${API}/api/stats`)).json();
    const eda = d.eda || {};
    const o = eda.overall || {};
    const seg = eda.worst_segment || {};

    const kpis = [
      ["Churn rate", (o.churn_rate_pct ?? 0).toFixed(1) + "%", true],
      ["Customers", (o.customers ?? 0).toLocaleString()],
      ["Monthly revenue at risk", money(o.mrr_at_risk ?? 0), true],
      ["Avg tenure, churned", (o.avg_tenure_churned ?? 0).toFixed(1) + " mo"],
      ["Avg tenure, retained", (o.avg_tenure_retained ?? 0).toFixed(1) + " mo"],
      [
        `Worst segment · ${seg.contract ?? "—"}, ${seg.tenure_group ?? "—"} mo`,
        (seg.churn_pct ?? 0).toFixed(1) + "%",
        true,
      ],
    ];
    $("kpis").innerHTML = kpis
      .map(
        ([k, v, accent]) =>
          `<div class="kpi ${accent ? "accent" : ""}"><span class="v">${v}</span><span class="k">${k}</span></div>`
      )
      .join("");

    const rows = (obj, unit = "%") =>
      Object.entries(obj || {})
        .map(
          ([k, v]) => `<div class="bar-row">
            <div class="top"><span>${k}</span><span>${(+v).toFixed(1)}${unit}</span></div>
            <div class="bar-track"><div class="bar-fill" data-w="${Math.min(100, +v)}"></div></div>
          </div>`
        )
        .join("");

    $("contract-bars").innerHTML = rows((eda.churn_by_category || {}).contract);

    const shap = (d.shap_importance || []).slice(0, 8);
    const top = shap.length ? shap[0].mean_abs_shap : 1;
    $("shap-bars").innerHTML = shap
      .map(
        (s) => `<div class="bar-row">
          <div class="top"><span>${s.feature}</span><span>${s.mean_abs_shap.toFixed(3)}</span></div>
          <div class="bar-track"><div class="bar-fill" data-w="${(s.mean_abs_shap / top) * 100}"></div></div>
        </div>`
      )
      .join("");

    const comp = d.model_comparison || [];
    if (comp.length) {
      const cols = ["model", "recall", "precision", "f1", "roc_auc"];
      const head = `<tr>${cols
        .map((c) => `<th>${c.replace("_", " ")}</th>`)
        .join("")}</tr>`;
      const best = META?.model;
      const body = comp
        .map(
          (r) =>
            `<tr class="${r.model === best ? "best" : ""}">${cols
              .map((c) => `<td>${typeof r[c] === "number" ? r[c].toFixed(3) : r[c]}</td>`)
              .join("")}</tr>`
        )
        .join("");
      $("model-table").querySelector("thead").innerHTML = head;
      $("model-table").querySelector("tbody").innerHTML = body;
    }

    $("figures").innerHTML = FIGURES.map(
      ([f, cap]) =>
        `<div class="fig"><img loading="lazy" src="${API}/api/figures/${f}" alt="${cap}"><p>${cap}</p></div>`
    ).join("");

    requestAnimationFrame(() =>
      document.querySelectorAll(".bar-fill").forEach((b) => (b.style.width = b.dataset.w + "%"))
    );
  } catch {
    $("kpis").innerHTML = `<div class="err">Insights need the API running. Start the backend and reload.</div>`;
  }
}

/* ------------------------------------------------------------------ */
/* wiring                                                              */
/* ------------------------------------------------------------------ */
function init() {
  buildAddons();
  syncInternetLock();
  loadHealth();
  loadSamples();

  $("churn-form").addEventListener("submit", score);
  document
    .querySelector('[name="internet_service"]')
    .addEventListener("change", syncInternetLock);

  $("reset-btn").addEventListener("click", () => {
    $("churn-form").reset();
    document.querySelectorAll(".addon").forEach((a) => a.classList.remove("on"));
    syncInternetLock();
    document.querySelector(".err")?.remove();
    $("result").hidden = true;
    $("result-empty").hidden = false;
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => {
        t.classList.remove("is-active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("is-active");
      tab.setAttribute("aria-selected", "true");
      const insights = tab.dataset.view === "insights";
      $("view-score").hidden = insights;
      $("view-insights").hidden = !insights;
      if (insights) loadInsights();
    });
  });
}

document.addEventListener("DOMContentLoaded", init);
