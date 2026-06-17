"""Visualizer nativo — webcam OpenCV, zero browser, salvataggio locale.

Uso:
    python -m mindruntime.visualizer
    ORGANISM-Windows.exe
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
import time

import numpy as np

from mindruntime.cuda_util import cuda_info
from mindruntime.local_store import append_tick, save_snapshot, store_info

RENDER_MODES = ("phase_coherence", "voltage", "impulse")


def _engine_cls():
    if os.environ.get("ORGANISM_ENGINE", "v2").lower() in ("legacy", "v1", "dendritic"):
        from mindruntime.dendritic_engine import DendriticBrainEngine
        return DendriticBrainEngine
    from mindruntime.gpu_engine_v2 import BrainEngineV2
    return BrainEngineV2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ORGANISM · cervello locale (no browser)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--image", type=str, default="")
    p.add_argument("--steps-per-frame", type=int, default=2)
    p.add_argument("--save-every", type=int, default=120, help="tick tra salvataggi automatici")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--legacy", action="store_true", help="motore dendritico V1")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--render-mode",
        choices=RENDER_MODES,
        default="phase_coherence",
        help="modalità rendering V2 (M per ciclare in runtime)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        import cv2
    except ImportError:
        print("opencv-python richiesto: pip install opencv-python", file=sys.stderr)
        return 1

    args = _parse_args(argv)
    info = cuda_info()
    paths = store_info()
    print("ORGANISM · solo locale (nessun browser, nessun server)")
    print(f"  Backend: {'CUDA' if info.get('cuda') else 'CPU'}")
    print(f"  Griglia: {args.width}×{args.height}")
    print(f"  Dati: {paths['dir']}")
    print("  Q/ESC esci · M modalità rendering · S snapshot\n")

    if args.legacy:
        os.environ["ORGANISM_ENGINE"] = "legacy"
    Engine = _engine_cls()
    engine = Engine(width=args.width, height=args.height)
    cap = None
    static_bgr: np.ndarray | None = None
    frames = 0
    t0 = time.perf_counter()
    last_save_tick = 0
    render_mode = args.render_mode
    render_idx = RENDER_MODES.index(render_mode) if render_mode in RENDER_MODES else 0

    def _flush_on_exit() -> None:
        if frames > 0:
            elapsed = max(1e-6, time.perf_counter() - t0)
            p = save_snapshot(engine, frames=frames, fps=frames / elapsed)
            print(f"Salvato: {p}")

    atexit.register(_flush_on_exit)

    if args.image:
        static_bgr = cv2.imread(args.image)
        if static_bgr is None:
            print(f"Immagine non trovata: {args.image}", file=sys.stderr)
            return 1
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Webcam non disponibile.", file=sys.stderr)
            return 1

    win = "ORGANISM · V2 fisica emergente"
    try:
        while True:
            if static_bgr is not None:
                bgr = static_bgr
            else:
                ok, bgr = cap.read()
                if not ok:
                    break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for _ in range(max(1, args.steps_per_frame)):
                result = engine.step(rgb)

            frames += 1

            if engine.stats.conscious or engine.stats.locked_symbol:
                append_tick(
                    {
                        "tick": engine.stats.tick,
                        "conscious": engine.stats.conscious,
                        "order": engine.stats.order_parameter,
                        "symbol": engine.stats.locked_symbol,
                        "recognition": result.get("recognition", []),
                    }
                )

            if engine.stats.tick - last_save_tick >= args.save_every:
                elapsed = max(1e-6, time.perf_counter() - t0)
                save_snapshot(engine, frames=frames, fps=frames / elapsed)
                last_save_tick = engine.stats.tick

            if not args.no_display:
                if hasattr(engine, "render") and render_mode in RENDER_MODES:
                    try:
                        brain_rgb = engine.render(mode=render_mode)
                        show = cv2.cvtColor(brain_rgb, cv2.COLOR_RGB2BGR)
                    except TypeError:
                        show = engine.render_composite(bgr)
                else:
                    show = engine.render_composite(bgr)

                zones = engine.get_recognition_zones() if hasattr(engine, "get_recognition_zones") else []
                for i, (zy, zx, coh) in enumerate(zones[:5]):
                    cv2.circle(show, (zx, zy), 5, (0, 255, 0), 2)

                y = 22
                for line in engine.overlay_lines():
                    cv2.putText(
                        show,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (220, 255, 200),
                        1,
                        cv2.LINE_AA,
                    )
                    y += 18
                cv2.putText(
                    show,
                    f"mode={render_mode} · save: {paths['dir'][-28:]}",
                    (10, show.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (160, 200, 160),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(win, show)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("m"), ord("M")):
                    render_idx = (render_idx + 1) % len(RENDER_MODES)
                    render_mode = RENDER_MODES[render_idx]
                    print(f"[INFO] Modalità rendering: {render_mode}")
                if key in (ord("s"), ord("S")) and hasattr(engine, "export_state"):
                    snap = engine.export_state()
                    fname = paths["dir"] + f"/brain_step_{engine.stats.tick}.npz"
                    np.savez_compressed(fname, **{k: v for k, v in snap.items() if k != "params"})
                    print(f"[INFO] Snapshot: {fname}")

            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    fps = frames / elapsed if elapsed > 0 else 0.0
    print(f"Chiuso: {frames} frame · {fps:.1f} FPS · R={engine.stats.order_parameter:.3f}")
    if engine.stats.last_recognition:
        print("  Simboli:", ", ".join(f"{s}:{sc:.2f}" for s, sc in engine.stats.last_recognition[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
