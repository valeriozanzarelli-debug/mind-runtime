"""Mindruntime — motore GPU locale (Numba CUDA) per simulazione cerebrale emergente."""

from mindruntime.dendritic_engine import DendriticBrainEngine
from mindruntime.gpu_engine import GPUBrainEngine
from mindruntime.gpu_engine_v2 import BrainEngineV2, SupremeBrainEngine

__all__ = ["DendriticBrainEngine", "GPUBrainEngine", "BrainEngineV2", "SupremeBrainEngine", "__version__"]
__version__ = "0.9.0"
