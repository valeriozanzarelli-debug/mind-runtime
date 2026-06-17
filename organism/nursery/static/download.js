/* ORGANISM — pagina download Windows */

const BASE = (window.ORGANISM_BASE || "").replace(/\/$/, "");

async function githubAssetReady(url) {
  if (!url) return false;
  try {
    const r = await fetch(url, { method: "HEAD", mode: "no-cors" });
    return r.ok || r.type === "opaque";
  } catch {
    return false;
  }
}

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

    btn.classList.add("disabled");
    btn.href = "#";

    if (w.local && w.url) {
      btn.href = w.url;
      btn.classList.remove("disabled");
      sizeLine.textContent = w.size_mb ? `~${w.size_mb} MB` : "";
      statusLine.textContent = "Download diretto da inkconscius.eu";
      return;
    }

    const gh = w.mirror_github || "";
    const ghReady = gh ? await githubAssetReady(gh) : false;
    if (ghReady) {
      btn.href = gh;
      btn.classList.remove("disabled");
      btn.querySelector(".btn-title").textContent = "Scarica ORGANISM";
      statusLine.textContent = "Download da GitHub (build ufficiale)";
      return;
    }

    statusLine.innerHTML =
      'Build in corso — l’.exe sarà disponibile tra pochi minuti. ' +
      (w.build_url
        ? `<a href="${w.build_url}" target="_blank" rel="noopener">Vedi stato build</a>`
        : "");
    statusLine.classList.add("warn");
    btn.querySelector(".btn-title").textContent = "Presto disponibile";
  } catch (err) {
    console.error(err);
    statusLine.textContent = "Pagina download non raggiungibile — il server va aggiornato.";
    statusLine.classList.add("warn");
    btn.querySelector(".btn-title").textContent = "Non disponibile";
  }
}

loadDownload();
