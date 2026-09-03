"""
Models package for Alzheimer's Disease Detection System.
"""

from .cnn_2d import EfficientNetB02D, ResNet182D, aggregate_slice_probabilities

__all__ = ["EfficientNetB02D", "ResNet182D", "aggregate_slice_probabilities"]
