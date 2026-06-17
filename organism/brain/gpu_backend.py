"""Backend GPU — PyTorch/CUDA su Windows/Linux, fallback CPU/numpy."""

from __future__ import annotations

from typing import Any

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    HAS_TORCH = False


def has_torch() -> bool:
    return HAS_TORCH


def cuda_available() -> bool:
    return HAS_TORCH and bool(torch.cuda.is_available())


def resolve_device(requested: str | None = "auto") -> str:
    """Risolve device: auto | cuda | cuda:0 | cpu | numpy."""
    req = (requested or "auto").strip().lower()
    if req == "numpy":
        return "numpy"
    if not HAS_TORCH:
        return "numpy"
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA non disponibile. Su Windows NVIDIA installa PyTorch GPU:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
        return req
    if req == "cpu":
        return "cpu"
    raise ValueError(f"device sconosciuto: {requested!r}")


def torch_device(name: str) -> Any:
    if not HAS_TORCH:
        raise RuntimeError("torch non installato")
    if name.startswith("cuda"):
        return torch.device(name)
    return torch.device("cpu")


def gpu_info() -> dict[str, Any]:
    """Info per diagnostica locale (Windows: verifica che la GPU sia visibile)."""
    info: dict[str, Any] = {
        "torch": False,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "recommended": "numpy",
    }
    if not HAS_TORCH:
        info["install_hint"] = (
            "pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
        return info
    info["torch"] = True
    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["recommended"] = "cuda"
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(props.total_memory / (1024**3), 2)
    else:
        info["recommended"] = "cpu"
        info["install_hint"] = (
            "GPU NVIDIA non rilevata. Installa driver + PyTorch CUDA, oppure usa --device cpu"
        )
    return info


def scalar(value: Any) -> float:
    """Estrae float da numpy scalar, torch tensor, o Python float."""
    if HAS_TORCH and torch is not None and isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
