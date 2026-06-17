"""Visualizer locale — webcam OpenCV, nessun server HTTP.

Uso:
    python -m mindruntime.visualizer
    python -m mindruntime.visualizer --image path/to/file.jpg
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from mindruntime.cuda_util import cuda_info
from mindruntime.gpu_engine import GPUBrainEngine


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mindruntime GPU brain — solo locale")
    p.add_argument("--width", type=int, default=256, help="larghezza griglia neuroni")
    p.add_argument("--height", type=int, default=256, help="altezza griglia neuroni")
    p.add_argument("--camera", type=int, default=0, help="indice webcam")
    p.add_argument("--image", type=str, default="", help="immagine statica al posto della webcam")
    p.add_argument("--steps-per-frame", type=int, default=2, help="tick fisici per frame")
    p.add_argument("--no-display", action="store_true", help="benchmark senza finestra")
    p.add_argument("--max-frames", type=int, default=0, help="0 = infinito")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    info = cuda_info()
    print("Mindruntime · motore GPU locale")
    print(f"  Numba CUDA: {info.get('cuda', False)} · device: {info.get('device', 'cpu')}")
    print(f"  Griglia: {args.width}×{args.height} (~{args.width * args.height // 1000}k neuroni)")
    print("  Nessun server — tutto sul tuo PC.\n")

    engine = GPUBrainEngine(width=args.width, height=args.height)
    use_cv2 = False
    cap = None
    static_img: np.ndarray | None = None

    if args.image:
        try:
            import cv2

            static_img = cv2.imread(args.image)
            if static_img is None:
                print(f"Impossibile leggere immagine: {args.image}", file=sys.stderr)
                return 1
            static_img = cv2.cvtColor(static_img, cv2.COLOR_BGR2RGB)
            use_cv2 = True
        except ImportError:
            from PIL import Image

            static_img = np.array(Image.open(args.image).convert("RGB"))
    else:
        try:
            import cv2

            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                print("Webcam non disponibile. Prova --image file.jpg", file=sys.stderr)
                return 1
            use_cv2 = True
        except ImportError:
            print("Installa opencv-python: pip install opencv-python", file=sys.stderr)
            return 1

    frames = 0
    t_start = time.perf_counter()
    try:
        while True:
            if static_img is not None:
                frame = static_img
            else:
                assert cap is not None
                ok, bgr = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            for _ in range(max(1, args.steps_per_frame)):
                engine.step(frame)

            frames += 1
            if not args.no_display:
                overlay = engine.render_overlay()
                bgr_show = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imshow("mindruntime — fase neuroni", bgr_show)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        if cap is not None:
            cap.release()
        if use_cv2:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass

    elapsed = time.perf_counter() - t_start
    fps = frames / elapsed if elapsed > 0 else 0.0
    rec = engine.stats.last_recognition
    print(f"Fine: {frames} frame · {fps:.1f} FPS medio · backend={engine.stats.backend}")
    if rec:
        print("  Riconoscimenti:", ", ".join(f"{s}:{sc:.2f}" for s, sc in rec[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
