"""CLI di CEREBRUM."""
from __future__ import annotations

import argparse
import os

from cerebrum import DEFAULT_PORT, __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="cerebrum",
        description="CEREBRUM — runtime cerebrale locale (GPU).",
    )
    parser.add_argument("command", nargs="?", default="serve",
                        choices=["serve", "info", "selftest"],
                        help="serve: avvia il cervello + server HTTP")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("CEREBRUM_PORT", DEFAULT_PORT)))
    parser.add_argument("--neurons", type=int,
                        default=int(os.environ.get("CEREBRUM_NEURONS", 4096)))
    args = parser.parse_args(argv)

    if args.command == "info":
        from cerebrum.neuro import describe_backend
        print(f"CEREBRUM v{__version__}")
        print("Backend:", describe_backend())
        return 0

    if args.command == "selftest":
        from cerebrum.brain import BrainConfig, Cerebrum
        b = Cerebrum(BrainConfig(neurons=512))
        b.birth()
        import time
        time.sleep(1.0)
        b.hear("ciao piccolo")
        time.sleep(0.5)
        print("health:", b.health())
        print("thought:", b.stream.current())
        b.shutdown()
        print("selftest OK")
        return 0

    from cerebrum.server import serve
    serve(port=args.port, neurons=args.neurons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
