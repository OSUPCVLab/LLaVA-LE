#!/usr/bin/env python3
"""
Main script to run GPT-4 Vision captioning for lunar surface imagery.

This script processes lunar multimodal datasets and generates scientific captions
using OpenAI's GPT-4 Vision API. It supports various OpenAI models, custom temperature
settings, dataset splitting, and flexible output options.

Usage:
    python main.py [options]

Prerequisites:
    - Set OpenAI API key: export OPENAI_API_KEY="your_key_here"
    - Ensure dataset structure: data/lumina/{tile_name}/{modality}/r{XX}_c{YY}.png

Basic Examples:
    # Process entire dataset with default settings (gpt-4o, temperature=0.3)
    python main.py

    # Process specific split with sample limit
    python main.py --split train --max-samples 100

    # Use custom model and temperature for more creative captions
    python main.py --model gpt-4o-mini --temperature 0.7

    # Deterministic output for reproducible results
    python main.py --temperature 0.0 --max-samples 50

Advanced Examples:
    # Custom dataset location and output directory
    python main.py --root-dir /path/to/lunar/data --output-dir /path/to/results

    # Process validation set with specific model
    python main.py --split val --model gpt-4-turbo --max-samples 500

    # High creativity for diverse caption generation
    python main.py --temperature 1.2 --model gpt-4o --split test

    # Use custom splits file and API key
    python main.py --splits-file custom_splits.json --api-key sk-your-key-here

    # Random sampling for sanity checking
    python main.py --random-sampling --max-samples 50 --seed 123

Model Management:
    # List all available OpenAI models
    python main.py --list-models

    # Use specific vision model
    python main.py --model gpt-4-vision-preview --temperature 0.5

Output:
    - Generates JSON files with captions, metadata, and location info
    - Files saved to outputs/ directory (or custom --output-dir)
    - Naming: lunar_captions_{split}_{timestamp}.json

Temperature Guidelines:
    - 0.0: Deterministic, consistent output
    - 0.3: Balanced (default), good for scientific accuracy
    - 0.7: More creative, varied descriptions
    - 1.0+: Highly creative, potentially inconsistent
"""

import os
import argparse
import sys
import json
import random
from datetime import datetime
from pathlib import Path

from curate.caption import LunarCaptionGenerator, LunarMultimodalDataset


def main():
    """Main function to run the caption generation."""
    parser = argparse.ArgumentParser(
        description="Generate scientific captions for lunar surface imagery using GPT-4 Vision"
    )

    parser.add_argument(
        "--root-dir",
        default="data/lumina",
        help="Root directory of the dataset (default: data/lumina)",
    )

    parser.add_argument(
        "--splits-file",
        default="data/splits.json",
        help="Path to splits JSON file (default: data/splits.json)",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        help="Specific split to process (train, val, test). If not specified, processes entire dataset.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process. If not specified, processes all samples.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save output files (default: outputs)",
    )

    parser.add_argument(
        "--api-key",
        help="OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable.",
    )

    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use for caption generation (default: gpt-4o)",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available OpenAI models and exit",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
        help="Temperature for text generation (0.0-2.0, default: 0.25)",
    )

    parser.add_argument(
        "--random-sampling",
        action="store_true",
        help="Randomly sample from dataset instead of sequential processing. Useful for sanity checking.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible random sampling (default: 42)",
    )

    args = parser.parse_args()

    # Validate temperature
    if not 0.0 <= args.temperature <= 2.0:
        print("Error: Temperature must be between 0.0 and 2.0")
        return 1

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        print(
            "Error: Please provide OpenAI API key using --api-key or set OPENAI_API_KEY environment variable"
        )
        return 1

    # Resolve paths relative to the dataset directory
    dataset_dir = Path(__file__).parent
    root_dir = dataset_dir / args.root_dir
    splits_file = dataset_dir / args.splits_file if args.splits_file else None
    output_dir = dataset_dir / args.output_dir

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if dataset directory exists
    if not root_dir.exists():
        print(f"Error: Dataset directory not found: {root_dir}")
        print("Please check the path or use --root-dir to specify the correct path")
        return 1

    # Check splits file if specified
    if splits_file and not splits_file.exists():
        print(f"Warning: Splits file not found: {splits_file}")
        print("Continuing without splits (will process entire dataset)")
        splits_file = None

    print("=" * 60)
    print("Lunar Surface Caption Generator")
    print("=" * 60)
    print(f"Dataset directory: {root_dir}")
    print(f"Splits file: {splits_file if splits_file else 'None'}")
    print(f"Split: {args.split if args.split else 'All'}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"Sampling: {'Random' if args.random_sampling else 'Sequential'}")
    if args.random_sampling:
        print(f"Random seed: {args.seed}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Initialize caption generator
    try:
        generator = LunarCaptionGenerator(api_key=api_key)

        # Handle list models option
        if args.list_models:
            print("Fetching available models from OpenAI API...")
            models = generator.get_available_models()
            print(f"\nAvailable models ({len(models)} total):")
            print("=" * 50)

            # Separate vision-capable models
            vision_models = [
                m
                for m in models
                if any(pattern in m for pattern in ["gpt-4", "gpt-4o", "gpt-4-vision"])
            ]
            other_models = [m for m in models if m not in vision_models]

            if vision_models:
                print("Vision-capable models (recommended for captions):")
                for model in vision_models:
                    print(f"  - {model}")
                print()

            if other_models:
                print("Other available models:")
                for model in other_models[:10]:  # Limit to first 10 to avoid spam
                    print(f"  - {model}")
                if len(other_models) > 10:
                    print(f"  ... and {len(other_models) - 10} more")

            return 0

        # Validate model
        if not generator.validate_model(args.model):
            return 1

        print(f"Using model: {args.model}")

    except Exception as e:
        print(f"Error initializing caption generator: {e}")
        return 1

    # Process dataset
    try:
        # Create dataset instance
        dataset = LunarMultimodalDataset(
            root_dir=str(root_dir),
            split=args.split,
            splits_file=str(splits_file) if splits_file else None,
        )

        print(f"Dataset initialized with {len(dataset)} samples")

        if len(dataset) == 0:
            print("No samples found in dataset. Please check your dataset structure.")
            return 1

        # Process samples
        total_samples = len(dataset)
        num_samples = (
            min(args.max_samples, total_samples) if args.max_samples else total_samples
        )

        print(f"Processing {num_samples} samples from dataset (total: {total_samples})")

        # Generate sample indices based on sampling method
        # Random sampling of the dataset may be used for sanity checking purposes
        # Because if you sample sequentially it does not cover all the cases that there is 
        if args.random_sampling:
            random.seed(args.seed)
            sample_indices = random.sample(range(total_samples), num_samples)
            print(f"Random sampling enabled (seed: {args.seed})")
        else:
            sample_indices = list(range(num_samples))

        # Collect all results
        all_results = {
            "general_info": {
                "total_samples": total_samples,
                "processed_samples": num_samples,
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "temperature": args.temperature,
                "split": args.split if args.split else "all",
            },
            "captions": [],
        }

        # Process each sample with progress bar
        successful_count = 0
        failed_count = 0

        from tqdm import tqdm

        with tqdm(total=num_samples, desc="Generating captions", unit="sample") as pbar:
            for idx, sample_idx in enumerate(sample_indices):
                try:
                    # Get sample from dataset
                    sample = dataset[sample_idx]

                    # Extract PIL images
                    pancro_image = sample["pancro"]
                    gravity_image = sample["gravity"]
                    slope_image = sample["slope"]

                    # Create identifier from dataset index and tile info
                    tile_name, row_col = dataset.index[sample_idx]
                    identifier = f"{tile_name}_{row_col}"

                    # Generate caption
                    caption = generator.generate_caption_from_images(
                        pancro_image,
                        gravity_image,
                        slope_image,
                        args.model,
                        args.temperature,
                    )

                    if caption:
                        # Build result dictionary with metadata
                        result = {
                            "location": {
                                "tile_name": tile_name,
                                "row_col": row_col,
                            },
                            "caption": caption,
                            "metadata": {
                                "timestamp": datetime.now().isoformat(),
                                "word_count": len(caption.split()),
                            },
                        }

                        all_results["captions"].append(result)
                        successful_count += 1
                        pbar.set_postfix(
                            {
                                "success": successful_count,
                                "failed": failed_count,
                                "current": identifier,
                            }
                        )
                    else:
                        failed_count += 1
                        pbar.set_postfix(
                            {
                                "success": successful_count,
                                "failed": failed_count,
                                "current": f"FAILED: {identifier}",
                            }
                        )

                except Exception as e:
                    failed_count += 1
                    pbar.set_postfix(
                        {
                            "success": successful_count,
                            "failed": failed_count,
                            "current": f"ERROR (idx {sample_idx}): {str(e)[:50]}{'...' if len(str(e)) > 50 else ''}",
                        }
                    )
                    continue
                finally:
                    pbar.update(1)

        # Update successful count
        all_results["general_info"]["successful_captions"] = successful_count

        # Save results to JSON file
        output_filename = f"lunar_captions_{args.split if args.split else 'all'}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{output_filename}_{timestamp}"

        output_file = output_dir / f"{output_filename}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print("Caption generation completed!")
        print(f"Successfully generated: {successful_count}/{num_samples} captions")
        if failed_count > 0:
            print(f"Failed: {failed_count}/{num_samples} captions")
        print(f"Results saved to: {output_file}")
        return 0

    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Please check your dataset path and structure.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
