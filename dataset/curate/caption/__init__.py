"""
Caption module for lunar surface imagery.

This module provides functionality to generate scientific captions for lunar surface imagery
using OpenAI's GPT-4 Vision API.
"""

from .gpt import LunarCaptionGenerator
from .lunardataset import (
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
