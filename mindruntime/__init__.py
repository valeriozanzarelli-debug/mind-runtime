"""Mindruntime — motore GPU locale (Numba CUDA) per simulazione cerebrale emergente."""

from mindruntime.dendritic_engine import DendriticBrainEngine
from mindruntime.gpu_engine import GPUBrainEngine

__all__ = ["DendriticBrainEngine", "GPUBrainEngine", "__version__"]
__version__ = "0.3.0"
