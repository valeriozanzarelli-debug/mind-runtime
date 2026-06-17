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
    const d = await r.json();
    const w = d.windows || {};

    ver.textContent = d.version || "0.5.0";
    minWin.textContent = `${w.min_windows || "10"}+`;
    btnSub.textContent = w.filename || "ORGANISM-Windows.exe";

    if (w.url) {
      btn.href = w.url;
      btn.classList.remove("disabled");
      if (w.size_mb) {
        sizeLine.textContent = `~${w.size_mb} MB`;
      }
      if (w.local) {
        statusLine.textContent = "Download diretto da questo server";
      } else {
        statusLine.textContent = "Download da GitHub Releases";
        statusLine.classList.add("warn");
      }
    } else {
      btn.href = "#";
      btn.classList.add("disabled");
      statusLine.textContent =
        "Build in corso — controlla tra poco o scarica da GitHub Releases";
      statusLine.classList.add("warn");
      if (w.mirror_github) {
        btn.href = w.mirror_github;
        btn.classList.remove("disabled");
        btn.querySelector(".btn-title").textContent = "Scarica da GitHub";
      }
    }
  } catch (err) {
    console.error(err);
    statusLine.textContent = "Errore caricamento info download";
    statusLine.classList.add("warn");
  }
}

loadDownload();
