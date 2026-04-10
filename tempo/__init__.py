# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Luke Hindman
"""
TEMPO: A Stochastic Benchmarking Protocol for Evaluating Temporal Robustness in Spiking Neural Networks.

This package provides tools for generating the TEMPO dataset and benchmarking
spiking neural networks for temporal robustness.
"""

__version__ = "0.1.0"
__author__ = "Luke Hindman"

from tempo.dataset.generate_dataset import generate_dataset

__all__ = ["generate_dataset"]
