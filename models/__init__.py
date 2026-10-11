"""Core model datatypes and trainable model wrappers.

Keep this package import lightweight so dataset-preparation code can depend on
the shared dataclasses without pulling in the torch training stack.
"""

from .data_structures import TemporalGraphSample

__all__ = ["TemporalGraphSample"]
