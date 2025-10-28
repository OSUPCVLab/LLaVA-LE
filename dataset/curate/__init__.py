"""
Lunar Caption Generator Package

A Python package for generating scientific captions for lunar surface imagery using GPT-4 Vision.
"""

__version__ = "0.1.0"
__author__ = "LLaVA-LE Project"

# Import main functionality from caption module
from .caption import (
    LunarCaptionGenerator,
    LunarMultimodalDataset,
    PancroDataset,
    SlopeDataset,
    GravityDataset,
    create_split_datasets,
)

__all__ = [
    "LunarCaptionGenerator",
    "LunarMultimodalDataset",
    "PancroDataset",
    "SlopeDataset",
    "GravityDataset",
    "create_split_datasets",
]
