"""Substrato neurale: campo di neuroni su GPU (torch) o CPU (numpy)."""

from cerebrum.neuro.field import NeuralField, describe_backend

__all__ = ["NeuralField", "describe_backend"]
