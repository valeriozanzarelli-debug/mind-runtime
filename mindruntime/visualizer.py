"""Visualizer nativo — webcam OpenCV, cervello dendritico, zero browser.

Uso:
    python -m mindruntime.visualizer
    ORGANISM-Windows.exe   (stesso entry point via launcher)
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from mindruntime.cuda_util import cuda_info
from mindruntime.dendritic_engine import DendriticBrainEngine


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ORGANISM · cervello dendritico locale")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--image", type=str, default="")
    p.add_argument("--steps-per-frame", type=int, default=2)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        import cv2
    except ImportError:
        print("opencv-python richiesto: pip install opencv-python", file=sys.stderr)
        return 1

    args = _parse_args(argv)
    info = cuda_info()
    print("ORGANISM · fisica emergente (Turing · SOC · lock-in)")
    print(f"  Backend: {'CUDA' if info.get('cuda') else 'CPU'}")
    print(f"  Griglia: {args.width}×{args.height}")
    print("  Q o ESC per uscire — nessun browser.\n")

    engine = DendriticBrainEngine(width=args.width, height=args.height)
    cap = None
    static_bgr: np.ndarray | None = None

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

    win = "ORGANISM · coscienza emergente (Turing+SOC)"
    frames = 0
    t0 = time.perf_counter()
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
                engine.step(rgb)

            frames += 1
            if not args.no_display:
                show = engine.render_composite(bgr)
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
                cv2.imshow(win, show)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    fps = frames / elapsed if elapsed > 0 else 0.0
    print(f"Chiuso: {frames} frame · {fps:.1f} FPS · coerenza={engine.stats.mean_coherence:.3f}")
    if engine.stats.last_recognition:
        print("  Simboli:", ", ".join(f"{s}:{sc:.2f}" for s, sc in engine.stats.last_recognition[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
