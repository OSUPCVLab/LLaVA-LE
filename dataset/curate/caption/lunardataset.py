import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset


class UniModalLunarBaseDataset(Dataset, ABC):
    def __init__(
        self,
        root_dir,
        transform=None,
        split: Optional[str] = None,
        splits_file: Optional[Union[str, Path]] = None,
    ):
        """
        Base class for single-modality datasets.

        Args:
            root_dir (str): Root directory of the dataset (e.g., 'spacedata').
            transform (callable, optional): Transformation to apply to the modality images.
            split (str, optional): Which split to use ('train', 'val', 'test'). If None, use all data.
            splits_file (str or Path, optional): Path to JSON file containing dataset splits.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        # Load split information if provided
        self.allowed_tiles = None
        if split is not None and splits_file is not None:
            self.allowed_tiles = self._load_split_tiles(splits_file, split)

        self.index = self._build_index()

    def _load_split_tiles(self, splits_file: Union[str, Path], split: str) -> List[str]:
        """Load the list of tiles for the specified split."""
        splits_file = Path(splits_file)
        if not splits_file.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_file}")

        with open(splits_file, "r") as f:
            data = json.load(f)

        # Handle both new format (with metadata) and old format
        if "splits" in data:
            splits = data["splits"]
        else:
            splits = data

        if split not in splits:
            raise ValueError(
                f"Split '{split}' not found in splits file. Available: {list(splits.keys())}"
            )

        return splits[split]

    def _build_index(self):
        """Build a list of all (tile, row_col) tuples where data exists."""
        index = []
        for tile_dir in sorted(self.root_dir.iterdir()):
            if not tile_dir.is_dir():
                continue

            # Filter by allowed tiles if split is specified
            if (
                self.allowed_tiles is not None
                and tile_dir.name not in self.allowed_tiles
            ):
                continue

            modality_dir = tile_dir / self.get_modality_name()
            if not modality_dir.exists():
                continue

            for img_file in modality_dir.glob("r*_c*.png"):
                row_col = img_file.stem  # e.g., 'r00_c01'
                index.append((tile_dir.name, row_col))
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        tile_name, row_col = self.index[idx]
        tile_path = self.root_dir / tile_name

        # Load modality image
        modality_img_path = tile_path / self.get_modality_name() / f"{row_col}.png"
        modality_image = Image.open(modality_img_path).convert("RGB")
        if self.transform:
            modality_image = self.transform(modality_image)

        return {
            "image": modality_image,
        }

    @abstractmethod
    def get_modality_name(self):
        """Returns the name of the modality subfolder to load images from."""
        pass


class PancroDataset(UniModalLunarBaseDataset):
    def get_modality_name(self):
        return "pancro"

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        return {
            "pancro": out["image"],
        }


class SlopeDataset(UniModalLunarBaseDataset):
    def get_modality_name(self):
        return "slope"

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        return {
            "slope": out["image"],
        }


class GravityDataset(UniModalLunarBaseDataset):
    def get_modality_name(self):
        return "gravity"

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        return {
            "gravity": out["image"],
        }


class LunarMultimodalDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        split: Optional[str] = None,
        splits_file: Optional[Union[str, Path]] = None,
    ):
        """
        Lunar Multimodal Dataset

        Dataset class to load all modalities (pancro, slope, gravity) using composition of single-modality datasets.

        Args:
            root_dir (str): Root directory of the dataset (e.g., 'spacedata').
            transform (callable, optional): Transformation to apply to the modality images.
            split (str, optional): Which split to use ('train', 'val', 'test'). If None, use all data.
            splits_file (str or Path, optional): Path to JSON file containing dataset splits.
        """
        # Create individual dataset instances for each modality
        self.pancro_dataset = PancroDataset(
            root_dir=root_dir,
            transform=transform,
            split=split,
            splits_file=splits_file,
        )

        self.slope_dataset = SlopeDataset(
            root_dir=root_dir,
            transform=transform,
            split=split,
            splits_file=splits_file,
        )

        self.gravity_dataset = GravityDataset(
            root_dir=root_dir,
            transform=transform,
            split=split,
            splits_file=splits_file,
        )

        # Find common indices where all modalities are available
        self._build_common_index()

    @property
    def index(self):
        """Return the common index for compatibility with gpt.py"""
        return self.common_index

    def _build_common_index(self):
        """Build index of samples that exist in all three modality datasets."""
        pancro_indices = set(self.pancro_dataset.index)
        slope_indices = set(self.slope_dataset.index)
        gravity_indices = set(self.gravity_dataset.index)

        # Find intersection of all three datasets
        common_indices = pancro_indices & slope_indices & gravity_indices
        self.common_index = sorted(list(common_indices))

        # Create mapping from common index to individual dataset indices
        self.index_mapping = {}
        for i, common_idx in enumerate(self.common_index):
            self.index_mapping[i] = {
                "pancro": self.pancro_dataset.index.index(common_idx),
                "slope": self.slope_dataset.index.index(common_idx),
                "gravity": self.gravity_dataset.index.index(common_idx),
            }

    def __len__(self):
        return len(self.common_index)

    def __getitem__(self, idx):
        if idx >= len(self.common_index):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.common_index)}"
            )

        # Get the mapped indices for each modality dataset
        indices = self.index_mapping[idx]

        # Get data from each modality dataset
        pancro_sample = self.pancro_dataset[indices["pancro"]]
        slope_sample = self.slope_dataset[indices["slope"]]
        gravity_sample = self.gravity_dataset[indices["gravity"]]

        # Combine all modalities
        return {
            "pancro": pancro_sample["pancro"],
            "slope": slope_sample["slope"],
            "gravity": gravity_sample["gravity"],
        }


def create_split_datasets(
    dataset_class,
    root_dir: Union[str, Path],
    splits_file: Union[str, Path],
    transform=None,
) -> Dict[str, Dataset]:
    """
    Create dataset instances for all splits (train, val, test).

    Args:
        dataset_class: Dataset class to instantiate (e.g., LunarMultimodalDataset, PancroDataset)
        root_dir: Root directory of the dataset
        splits_file: Path to JSON file containing dataset splits
        transform: Transformation to apply to images

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing dataset instances
    """
    splits_file = Path(splits_file)
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = dataset_class(
            root_dir=root_dir,
            transform=transform,
            split=split,
            splits_file=splits_file,
        )

    return datasets


if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.ToTensor()

    # Example usage for a single modality
    pancro_dataset = PancroDataset(root_dir="data/lumina", transform=transform)
    sample = pancro_dataset[0]

    print("Pancro Dataset Sample:", sample["pancro"].size())

    # Example usage for all modalities
    multimodal_dataset = LunarMultimodalDataset(
        root_dir="data/lumina", transform=transform
    )
    sample = multimodal_dataset[0]
    print("Pancro image shape:", sample["pancro"].shape)
    print("Slope image shape:", sample["slope"].shape)
    print("Gravity image shape:", sample["gravity"].shape)

    # Sanity check: Visualize one sample image
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Convert tensors to numpy for grayscale images
    pancro_img = sample["pancro"].squeeze().numpy()
    slope_img = sample["slope"].squeeze().numpy()
    gravity_img = sample["gravity"].squeeze().numpy()

    axes[0].imshow(pancro_img, cmap="gray")
    axes[0].set_title("Pancro")

    axes[1].imshow(slope_img, cmap="gray")
    axes[1].set_title("Slope")

    axes[2].imshow(gravity_img, cmap="gray")
    axes[2].set_title("Gravity")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("lunar_multimodal_sanity_check.png")
