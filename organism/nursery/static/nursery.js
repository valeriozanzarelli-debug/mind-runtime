/* ORGANISM Nursery dashboard */

const LAYER_COLORS = {
  sensory: "#58a6ff",
  associative: "#3fb950",
  motor: "#f778ba",
};

let network = null;
let lastCycle = -1;

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  return r.json();
}

function el(id) {
  return document.getElementById(id);
}

async function refresh() {
  const state = await api("/api/state");
  updateStatus(state);
  renderThoughts(state.thought_stream || []);
  renderGrowthChart(state.growth || []);
  if (state.graph) renderGraph(state.graph);
  if (state.verification) renderVerify(state.verification);
}

function updateStatus(state) {
  const born = el("born-badge");
  born.textContent = state.born ? "vivo" : "non nato";
  born.className = "badge " + (state.born ? "on" : "off");
  el("phase-badge").textContent = state.phase || "—";
  const s = state.stats || {};
  el("stats-line").textContent = state.born
    ? `${s.neurons} neuroni · ${s.synapses} sinapsi · peso μ=${(s.mean_synapse_weight || 0).toFixed(4)} · cicli ${s.learning_cycles || 0}`
    : "Premi Nascita per dispiegare il DNA";
}

function renderThoughts(lines) {
  const box = el("thought-stream");
  box.innerHTML = lines
    .map((line) => {
      let cls = "thought-line";
      if (line.includes("stimolo:")) cls += " stim";
      else if (line.includes("→")) cls += " sym";
      else if (line.includes("⚡")) cls += " act";
      else if (line.includes("💬")) cls += " say";
      else if (line.includes("🧬")) cls += " new";
      return `<div class="${cls}">${escapeHtml(line)}</div>`;
    })
    .join("");
  box.scrollTop = box.scrollHeight;
}

function renderGraph(data) {
  const container = el("brain-graph");
  const nodes = new vis.DataSet(
    (data.nodes || []).map((n) => ({
      id: n.id,
      label: n.label,
      title: `${n.subtype}\nattivazione: ${n.activation}`,
      color: LAYER_COLORS[n.group] || "#888",
      size: 8 + n.activation * 20,
    }))
  );
  const edges = new vis.DataSet(
    (data.edges || []).map((e) => ({
      from: e.from,
      to: e.to,
      width: e.width || 1,
      title: `peso: ${e.weight}`,
      color: { opacity: 0.35 + e.weight * 0.5 },
    }))
  );
  const options = {
    physics: {
      stabilization: { iterations: 80 },
      barnesHut: { gravitationalConstant: -3000, springLength: 120 },
    },
    interaction: { hover: true, tooltipDelay: 100 },
    nodes: { font: { size: 10, color: "#ccc" } },
    edges: { smooth: { type: "continuous" } },
  };
  if (!network) {
    network = new vis.Network(container, { nodes, edges }, options);
  } else {
    network.setData({ nodes, edges });
  }
  const m = data.meta || {};
  el("graph-meta").textContent =
    `attivi: ${m.active_shown || 0}/${m.total_neurons || 0} · ` +
    `connessioni visibili: ${m.edges_shown || 0} · tick: ${m.tick || 0} · ` +
    `layer: ${JSON.stringify(m.layer_active || {})}`;
}

function renderGrowthChart(timeline) {
  const canvas = el("growth-chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#080c10";
  ctx.fillRect(0, 0, w, h);
  if (timeline.length < 2) {
    ctx.fillStyle = "#7d8b9a";
    ctx.font = "12px monospace";
    ctx.fillText("Insegna qualcosa per vedere la crescita…", 20, h / 2);
    return;
  }
  const weights = timeline.map((t) => t.mean_weight);
  const maxW = Math.max(...weights) * 1.05;
  const minW = Math.min(...weights) * 0.95;
  const pad = 30;
  ctx.strokeStyle = "#3fb950";
  ctx.lineWidth = 2;
  ctx.beginPath();
  timeline.forEach((t, i) => {
    const x = pad + (i / (timeline.length - 1)) * (w - pad * 2);
    const y = h - pad - ((t.mean_weight - minW) / (maxW - minW || 1)) * (h - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#7d8b9a";
  ctx.font = "10px monospace";
  ctx.fillText("peso medio sinapsi →", pad, 14);
  ctx.fillText(minW.toFixed(4), 4, h - pad);
  ctx.fillText(maxW.toFixed(4), 4, pad + 4);
}

function renderVerify(v) {
  const panel = el("verify-panel");
  if (!v.checks) return;
  const ok = v.auto_development;
  panel.innerHTML =
    `<strong class="${ok ? "check-ok" : "check-fail"}">` +
    `Auto-sviluppo DNA: ${ok ? "VERIFICATO ✓" : "IN CORSO…"}</strong><ul>` +
    v.checks
      .map(
        (c) =>
          `<li class="${c.ok ? "check-ok" : "check-fail"}">${c.id}: ${escapeHtml(c.detail)}</li>`
      )
      .join("") +
    "</ul>";
}

async function loadPhases() {
  const { phases } = await api("/api/phases");
  const list = el("phases-list");
  list.innerHTML = phases
    .map(
      (p) => `
    <div class="phase-block" data-phase="${p.id}">
      <h4>${escapeHtml(p.label)}</h4>
      <p>${escapeHtml(p.description)}</p>
      <button class="lesson-btn" onclick="runPhase('${p.id}')">Tutta la fase</button>
      ${p.lessons
        .map(
          (l) =>
            `<button class="lesson-btn" onclick="runLesson('${p.id}','${l.id}')">${escapeHtml(l.label)}</button>`
        )
        .join("")}
    </div>`
    )
    .join("");
}

window.runPhase = async (id) => {
  await api(`/api/phase/${id}`, { method: "POST", body: "{}" });
  await refresh();
};

window.runLesson = async (phaseId, lessonId) => {
  await api(`/api/phase/${phaseId}`, {
    method: "POST",
    body: JSON.stringify({ lesson: lessonId }),
  });
  await refresh();
};

el("btn-birth").onclick = async () => {
  await api("/api/birth", { method: "POST", body: "{}" });
  await refresh();
};

el("btn-curriculum").onclick = async () => {
  el("btn-curriculum").disabled = true;
  await api("/api/curriculum", { method: "POST", body: "{}" });
  el("btn-curriculum").disabled = false;
  await refresh();
};

el("btn-sleep").onclick = async () => {
  await api("/api/sleep", { method: "POST", body: "{}" });
  await refresh();
};

el("btn-verify").onclick = async () => {
  const v = await api("/api/verify");
  renderVerify(v);
};

el("btn-teach").onclick = async () => {
  const text = el("free-input").value.trim();
  if (!text) return;
  await api("/api/teach", {
    method: "POST",
    body: JSON.stringify({ input: { text }, modality: el("modality").value }),
  });
  await refresh();
};

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

loadPhases();
refresh();
setInterval(refresh, 800);
