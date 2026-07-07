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

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");
const port = process.env.CEREBRUM_PORT || "8788";
const isWin = process.platform === "win32";

function candidateExe() {
  const names = isWin ? ["cerebrum.exe", "CEREBRUM.exe"] : ["cerebrum"];
  for (const dir of [root, join(root, "bin"), join(root, "dist"), __dirname]) {
    for (const n of names) {
      const p = join(dir, n);
      if (existsSync(p)) return p;
    }
  }
  return null;
}

function launch() {
  const env = { ...process.env, CEREBRUM_PORT: String(port) };
  const exe = candidateExe();
  let child;
  if (exe) {
    console.log(`[cerebrum] avvio EXE: ${exe} (porta ${port})`);
    child = spawn(exe, ["serve", "--port", String(port)], { cwd: root, env, stdio: "inherit" });
  } else {
    const py = isWin ? "python" : (process.env.PYTHON || "python3");
    console.log(`[cerebrum] avvio: ${py} -m cerebrum serve --port ${port}`);
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
