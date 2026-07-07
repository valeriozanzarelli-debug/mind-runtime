#!/usr/bin/env node
// Launcher di compatibilità per Ink Admin.
// Ink Admin avvia `node scripts/serve_local.mjs`; questo script individua e
// lancia il runtime CEREBRUM vero e proprio (Python/GPU), inoltrando la porta.
//
// Ordine di ricerca:
//   1) EXE impacchettato accanto (cerebrum.exe / cerebrum) — build Windows
//   2) python -m cerebrum serve  (installazione da sorgente / venv)
//
// Tutto gira in locale: nessun processo su server remoto.

import { spawn, spawnSync } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");
const port = process.env.CEREBRUM_PORT || "8788";
const isWin = process.platform === "win32";

// Preferenza runtime: auto | python | exe (env CEREBRUM_RUNTIME).
// In "auto" usiamo Python SOLO se ha una GPU CUDA disponibile (per sfruttarla),
// altrimenti l'EXE impacchettato. Cosi', dopo SETUP_GPU_WINDOWS.ps1, lo stesso
// "Avvia" di Ink Admin gira sulla GPU senza altre modifiche.
const runtimePref = (process.env.CEREBRUM_RUNTIME || "auto").toLowerCase();

function pythonCmd() {
  return isWin ? "python" : (process.env.PYTHON || "python3");
}

function pythonHasCuda() {
  try {
    const r = spawnSync(pythonCmd(),
      ["-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
      { timeout: 20000 });
    return r.status === 0;
  } catch {
    return false;
  }
}

function pythonHasCerebrum() {
  try {
    const r = spawnSync(pythonCmd(), ["-c", "import cerebrum"], { timeout: 20000 });
    return r.status === 0;
  } catch {
    return false;
  }
}

function candidateExe() {
  // Solo su Windows cerchiamo l'EXE impacchettato. Su Linux/macOS il nome
  // "cerebrum" coincide con la cartella del package, quindi verifichiamo che
  // il candidato sia un FILE, non una directory.
  const names = isWin ? ["cerebrum.exe", "CEREBRUM.exe"] : ["cerebrum.bin"];
  for (const dir of [root, join(root, "bin"), join(root, "dist"), __dirname]) {
    for (const n of names) {
      const p = join(dir, n);
      try {
        if (existsSync(p) && statSync(p).isFile()) return p;
      } catch {}
    }
  }
  return null;
}

function decideMode() {
  const exe = candidateExe();
  if (runtimePref === "python") return { mode: "python" };
  if (runtimePref === "exe") return exe ? { mode: "exe", exe } : { mode: "python" };
  // auto: se python ha CUDA e il package cerebrum, usa la GPU via Python
  if (pythonHasCerebrum() && pythonHasCuda()) return { mode: "python", gpu: true };
  if (exe) return { mode: "exe", exe };
  return { mode: "python" };
}

function launch() {
  const env = { ...process.env, CEREBRUM_PORT: String(port) };
  const decision = decideMode();
  let child;
  if (decision.mode === "exe") {
    console.log(`[cerebrum] avvio EXE (CPU): ${decision.exe} (porta ${port})`);
    child = spawn(decision.exe, ["serve", "--port", String(port)], { cwd: root, env, stdio: "inherit" });
  } else {
    const py = pythonCmd();
    if (decision.gpu) {
      console.log(`[cerebrum] avvio su GPU: ${py} -m cerebrum serve --port ${port}`);
    } else {
      console.log(`[cerebrum] avvio: ${py} -m cerebrum serve --port ${port}`);
    }
    child = spawn(py, ["-m", "cerebrum", "serve", "--port", String(port)], {
      cwd: root, env, stdio: "inherit",
    });
  }

  child.on("error", (err) => {
    console.error("[cerebrum] impossibile avviare il runtime:", err.message);
    console.error("[cerebrum] installa Python 3.10+ e `pip install -e .` nella cartella, oppure usa la build Windows.");
    process.exit(1);
  });
  child.on("exit", (code) => process.exit(code ?? 0));

  const stop = () => { try { child.kill(); } catch {} };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
}

launch();
