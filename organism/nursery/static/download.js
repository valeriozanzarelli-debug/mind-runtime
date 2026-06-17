/* ORGANISM — pagina download Windows */

const BASE = (window.ORGANISM_BASE || "").replace(/\/$/, "");

async function loadDownload() {
  const ver = document.getElementById("ver");
  const minWin = document.getElementById("min-win");
  const btn = document.getElementById("btn-download");
  const btnSub = document.getElementById("btn-sub");
  const sizeLine = document.getElementById("size-line");
  const statusLine = document.getElementById("status-line");

  try {
    const r = await fetch(`${BASE}/api/download`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    const w = d.windows || {};

    ver.textContent = d.version || "0.5.0";
    minWin.textContent = `${w.min_windows || "10"}+`;
    btnSub.textContent = w.filename || "ORGANISM-Windows.exe";

    if (w.available && w.url) {
      btn.href = w.url;
      btn.classList.remove("disabled");
      btn.querySelector(".btn-title").textContent = "Scarica ORGANISM";
      sizeLine.textContent = w.size_mb ? `~${w.size_mb} MB` : "~27 MB";
      statusLine.textContent = w.local
        ? "Download diretto da inkconscius.eu"
        : "Download da GitHub (build ufficiale)";
      return;
    }

    btn.href = "#";
    btn.classList.add("disabled");
    btn.querySelector(".btn-title").textContent = "Presto disponibile";
    statusLine.innerHTML =
      "Build in corso — riprova tra qualche minuto. " +
      (w.build_url
        ? `<a href="${w.build_url}" target="_blank" rel="noopener">Vedi stato build</a>`
        : "");
    statusLine.classList.add("warn");
  } catch (err) {
    console.error(err);
    statusLine.textContent = "Pagina download non raggiungibile — aggiorna la pagina.";
    statusLine.classList.add("warn");
    document.getElementById("btn-download").classList.add("disabled");
  }
}

loadDownload();
