# Lunar Surface Caption Generator

This module generates scientific captions for lunar surface imagery using OpenAI's GPT-4 Vision API. It processes multimodal lunar datasets (panchromatic, gravity, slope images) and generates contextual captions with configurable creativity levels.

## Setup

1. Install required dependencies:
```bash
pip install openai torch torchvision pillow numpy matplotlib
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

## Usage

### Quick Start Examples

```bash
# Process entire dataset with default settings (gpt-4o, temperature=0.3)
python main.py

# Process 50 training samples for quick testing
python main.py --split train --max-samples 50

# Generate deterministic captions for reproducible results
python main.py --temperature 0.0 --max-samples 100

# Use more creative captions with higher temperature
python main.py --temperature 1.2 --model gpt-4o
```

### Model Management

```bash
# List all available OpenAI models
python main.py --list-models

# Use specific models for different purposes
python main.py --model gpt-4o-mini --temperature 0.5        # Faster, cost-effective
python main.py --model gpt-4-turbo --temperature 0.7        # High quality, balanced
python main.py --model gpt-4-vision-preview --temperature 0.3  # Latest vision model
```

### Practical Workflows

```bash
# Development workflow: Quick testing with small samples
python main.py --split train --max-samples 10 --output-dir dev_results

# Production workflow: Process validation set with optimal settings
python main.py --split val --model gpt-4o --temperature 0.3 --output-dir production

# Research workflow: Generate diverse captions for analysis
python main.py --temperature 1.0 --max-samples 500 --output-dir research_data

# Custom dataset workflow
python main.py --root-dir /path/to/custom/data --splits-file custom_splits.json --output-dir custom_results
```

### Command Line Arguments

- `--root-dir`: Root directory of the dataset (default: `data/lumina`)
- `--splits-file`: Path to splits JSON file (default: `data/splits.json`)
- `--split`: Specific split to process (`train`, `val`, `test`)
- `--max-samples`: Maximum number of samples to process
- `--output-dir`: Directory to save output files (default: `outputs`)
- `--api-key`: OpenAI API key (alternative to environment variable)
- `--model`: OpenAI model to use (default: `gpt-4o`)
- `--temperature`: Generation creativity (0.0-2.0, default: 0.3)
- `--list-models`: List available models and exit

### Temperature Settings Guide

Temperature controls caption creativity and determinism:

```bash
# Deterministic, consistent captions (good for reproducible research)
python main.py --temperature 0.0

# Slightly creative, balanced (default, recommended for most use cases)
python main.py --temperature 0.3

# Moderately creative captions (good for diverse training data)
python main.py --temperature 0.7

# Highly creative captions (experimental, may be less accurate)
python main.py --temperature 1.5
```

### Programmatic Usage

You can also use the caption generator directly:

```python
import os
from curate.caption.gpt import LunarCaptionGenerator
from lunardataset import LunarMultimodalDataset

# Initialize generator
generator = LunarCaptionGenerator()

# Load dataset
dataset = LunarMultimodalDataset(root_dir="data/lumina")
sample = dataset[0]

# Generate caption for single sample
caption = generator.generate_caption_from_images(
    sample["pancro"], 
    sample["gravity"], 
    sample["slope"],
    model="gpt-4o",
    temperature=0.3
)

print(f"Generated caption: {caption}")
```

## Dataset Structure

The expected dataset structure is:
```
data/lumina/
├── tile_1/
│   ├── pancro/
│   │   ├── r00_c00.png
│   │   └── ...
│   ├── gravity/
│   │   ├── r00_c00.png
│   │   └── ...
│   └── slope/
│       ├── r00_c00.png
│       └── ...
├── tile_2/
│   └── ...
└── ...
```

## Output

The script generates timestamped JSON files in the specified output directory with comprehensive metadata and captions.

### Output Structure

```json
{
  "general_info": {
    "total_samples": 1000,
    "processed_samples": 100,
    "timestamp": "2025-10-27T14:30:15.123456",
    "model": "gpt-4o",
    "temperature": 0.3,
    "split": "train",
    "successful_captions": 98
  },
  "captions": [
    {
      "identifier": "tile_001_r10_c20",
      "location": {
        "tile_name": "tile_001",
        "row_col": "r10_c20"
      },
      "caption": "This lunar surface region displays a complex topographical landscape characterized by numerous impact craters of varying sizes and depths. The panchromatic imagery reveals distinct albedo variations suggesting diverse geological compositions, while the gravity data indicates subsurface density anomalies consistent with buried crater structures.",
      "metadata": {
        "timestamp": "2025-10-27T14:30:22.456789",
        "word_count": 45
      }
    }
  ]
}
```

### Output File Naming

Files are automatically named with timestamps to prevent overwrites:
- `lunar_captions_train_20251027_143015.json` (for train split)
- `lunar_captions_all_20251027_143015.json` (for entire dataset)

### Practical Output Examples

```bash
# Generate captions for 10 samples and check results
python main.py --max-samples 10 --output-dir test_run
ls test_run/  # Shows: lunar_captions_all_20251027_143015.json

# Process validation set with specific model
python main.py --split val --model gpt-4-turbo --output-dir validation_results
# Outputs: validation_results/lunar_captions_val_20251027_143015.json
```