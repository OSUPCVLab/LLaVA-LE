#!/usr/bin/env python3
"""
Main script to run GPT-4 Vision captioning for lunar surface imagery.
Modified to save progress incrementally every N captions with proper resume functionality.
"""

import os
import argparse
import sys
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

from curate.caption import LunarCaptionGenerator, LunarMultimodalDataset


def find_latest_output_file(output_dir, split):
    """
    Find the most recent output file for the given split.
    
    Args:
        output_dir: Directory containing output files
        split: Split name (train/val/test/all)
        
    Returns:
        Path to latest file, or None if not found
    """
    pattern = f"lunar_captions_{split}_*.json"
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    # Find all matching files
    matching_files = list(output_path.glob(pattern))
    
    if not matching_files:
        return None
    
    # Return the most recently modified file
    latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def load_existing_progress(output_file):
    """
    Load existing progress from output file if it exists.
    
    Returns:
        tuple: (results_dict, processed_identifiers_set)
    """
    if output_file and output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # Extract identifiers of already processed samples
            processed = set()
            for caption_entry in existing_results.get('captions', []):
                location = caption_entry.get('location', {})
                identifier = f"{location.get('tile_name', '')}_{location.get('row_col', '')}"
                processed.add(identifier)
            
            print(f"✓ Found existing progress: {len(processed)} captions already generated")
            return existing_results, processed
        except Exception as e:
            print(f"Warning: Could not load existing progress: {e}")
            print("Starting fresh...")
    
    return None, set()


def save_results(output_file, results, create_backup=True):
    """
    Save results to JSON file with optional backup of previous version.
    
    Args:
        output_file: Path to output file
        results: Results dictionary to save
        create_backup: Whether to backup existing file before overwriting
    """
    try:
        # Create backup of existing file
        if create_backup and output_file.exists():
            backup_file = output_file.with_suffix('.json.backup')
            shutil.copy2(output_file, backup_file)
        
        # Write new results atomically (write to temp file, then rename)
        temp_file = output_file.with_suffix('.json.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Rename temp file to actual file (atomic operation)
        temp_file.replace(output_file)
        
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def main():
    """Main function to run the caption generation."""
    parser = argparse.ArgumentParser(
        description="Generate scientific captions for lunar surface imagery using GPT Vision API"
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
        help="OpenAI model to use for caption generation (default: gpt-4o). Examples: gpt-4o, gpt-4-turbo, gpt-5.1",
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
        help="Randomly sample from dataset instead of sequential processing.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible random sampling (default: 42)",
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save progress every N captions (default: 50)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if it exists",
    )

    parser.add_argument(
        "--output-name",
        help="Custom name for output file (without extension). If not provided, uses timestamp.",
    )

    args = parser.parse_args()

    # Validate temperature
    if not 0.0 <= args.temperature <= 2.0:
        print("Error: Temperature must be between 0.0 and 2.0")
        return 1

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("\n" + "=" * 60)
        print("ERROR: OpenAI API Key Required")
        print("=" * 60)
        print("\nYou need to provide your OpenAI API key in one of two ways:")
        print("\n1. Set as environment variable (RECOMMENDED):")
        print("   Windows (Command Prompt):")
        print("      set OPENAI_API_KEY=your_api_key_here")
        print("   Windows (PowerShell):")
        print("      $env:OPENAI_API_KEY='your_api_key_here'")
        print("   Linux/Mac:")
        print("      export OPENAI_API_KEY='your_api_key_here'")
        print("\n2. Pass directly via command line:")
        print("      python main.py --api-key your_api_key_here")
        print("\nTo get your API key, visit: https://platform.openai.com/api-keys")
        print("=" * 60 + "\n")
        return 1

    # Resolve paths - all relative to script location
    dataset_dir = Path(__file__).parent
    root_dir = Path(args.root_dir) if os.path.isabs(args.root_dir) else dataset_dir / args.root_dir
    splits_file = Path(args.splits_file) if args.splits_file else None
    if splits_file and not os.path.isabs(args.splits_file):
        splits_file = dataset_dir / args.splits_file
    output_dir = Path(args.output_dir) if os.path.isabs(args.output_dir) else dataset_dir / args.output_dir

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if dataset directory exists
    if not root_dir.exists():
        print(f"Error: Dataset directory not found: {root_dir}")
        print("Please check the path or use --root-dir to specify the correct path")
        print(f"Expected structure: {root_dir}/tile_name/pancro|gravity|slope/")
        return 1

    # Check splits file if specified
    if splits_file and not splits_file.exists():
        print(f"Warning: Splits file not found: {splits_file}")
        print("Continuing without splits (will process entire dataset)")
        splits_file = None

    print("=" * 60)
    print("Lunar Surface Caption Generator (Incremental Save Mode)")
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
    print(f"Save interval: Every {args.save_interval} captions")
    print(f"Resume mode: {'Enabled' if args.resume else 'Disabled'}")
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
                if any(pattern in m for pattern in ["gpt-4", "gpt-5", "gpt-4o", "gpt-4-vision", "vision"])
            ]
            other_models = [m for m in models if m not in vision_models]

            if vision_models:
                print("Vision-capable models (recommended for captions):")
                for model in vision_models:
                    print(f"  - {model}")
                print()

            if other_models:
                print("Other available models:")
                for model in other_models[:10]:
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

        # Determine output file name
        split_name = args.split if args.split else 'all'
        
        # Check for resume mode
        output_file = None
        processed_identifiers = set()
        all_results = None
        successful_count = 0
        
        if args.resume:
            # Try to find existing file
            output_file = find_latest_output_file(output_dir, split_name)
            
            if output_file:
                print(f"Found existing file for resume: {output_file.name}")
                all_results, processed_identifiers = load_existing_progress(output_file)
                if all_results:
                    successful_count = len(all_results['captions'])
                    print(f"Resuming from {successful_count} existing captions")
                else:
                    output_file = None
        
        # If no existing file found or not resuming, create new filename
        if output_file is None:
            if args.output_name:
                output_filename = f"lunar_captions_{split_name}_{args.output_name}"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"lunar_captions_{split_name}_{timestamp}"
            
            output_file = output_dir / f"{output_filename}.json"
            print(f"Creating new output file: {output_file.name}")

        # Initialize results structure if needed
        total_samples = len(dataset)
        num_samples = (
            min(args.max_samples, total_samples) if args.max_samples else total_samples
        )

        if all_results is None:
            all_results = {
                "general_info": {
                    "total_samples": total_samples,
                    "processed_samples": num_samples,
                    "timestamp": datetime.now().isoformat(),
                    "model": args.model,
                    "temperature": args.temperature,
                    "split": split_name,
                },
                "captions": [],
            }

        print(f"Processing {num_samples} samples from dataset (total: {total_samples})")

        # Generate sample indices based on sampling method
        if args.random_sampling:
            random.seed(args.seed)
            sample_indices = random.sample(range(total_samples), num_samples)
            print(f"Random sampling enabled (seed: {args.seed})")
        else:
            sample_indices = list(range(num_samples))

        failed_count = 0
        captions_since_last_save = 0
        skipped_count = len(processed_identifiers)

        from tqdm import tqdm

        with tqdm(total=num_samples, desc="Generating captions", unit="sample", initial=skipped_count) as pbar:
            for idx, sample_idx in enumerate(sample_indices):
                try:
                    # Get sample from dataset
                    sample = dataset[sample_idx]

                    # Create identifier from dataset index and tile info
                    tile_name, row_col = dataset.index[sample_idx]
                    identifier = f"{tile_name}_{row_col}"

                    # Skip if already processed (in resume mode)
                    if identifier in processed_identifiers:
                        pbar.update(1)
                        continue

                    # Extract PIL images
                    pancro_image = sample["pancro"]
                    gravity_image = sample["gravity"]
                    slope_image = sample["slope"]

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
                        processed_identifiers.add(identifier)
                        successful_count += 1
                        captions_since_last_save += 1

                        # Save progress every N captions
                        if captions_since_last_save >= args.save_interval:
                            all_results["general_info"]["successful_captions"] = successful_count
                            all_results["general_info"]["last_updated"] = datetime.now().isoformat()
                            
                            if save_results(output_file, all_results):
                                pbar.write(f"✓ Progress saved: {successful_count} captions ({output_file.name})")
                                captions_since_last_save = 0
                            else:
                                pbar.write(f"✗ Failed to save progress at {successful_count} captions")

                        pbar.set_postfix(
                            {
                                "success": successful_count,
                                "failed": failed_count,
                                "unsaved": captions_since_last_save,
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
                    error_msg = str(e)[:50] + ('...' if len(str(e)) > 50 else '')
                    pbar.set_postfix(
                        {
                            "success": successful_count,
                            "failed": failed_count,
                            "current": f"ERROR: {error_msg}",
                        }
                    )
                    continue
                finally:
                    pbar.update(1)

        # Final save with all remaining captions
        all_results["general_info"]["successful_captions"] = successful_count
        all_results["general_info"]["failed_captions"] = failed_count
        all_results["general_info"]["completion_timestamp"] = datetime.now().isoformat()

        if save_results(output_file, all_results, create_backup=False):
            print(f"\n{'=' * 60}")
            print("Caption generation completed!")
            print(f"Successfully generated: {successful_count}/{num_samples} captions")
            if skipped_count > 0:
                print(f"Skipped (already done): {skipped_count} captions")
            if failed_count > 0:
                print(f"Failed: {failed_count}/{num_samples} captions")
            print(f"Final results saved to: {output_file}")
            
            # Clean up backup file if it exists
            backup_file = output_file.with_suffix('.json.backup')
            if backup_file.exists():
                backup_file.unlink()
                print(f"Backup file removed: {backup_file.name}")
        else:
            print(f"\n{'=' * 60}")
            print("Warning: Final save failed!")
            print(f"Check the backup file: {output_file.with_suffix('.json.backup')}")

        return 0

    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Please check your dataset path and structure.")
        
        # Try to save whatever progress we have
        if 'all_results' in locals() and all_results:
            emergency_file = output_dir / f"emergency_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if save_results(emergency_file, all_results, create_backup=False):
                print(f"Emergency save successful: {emergency_file}")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)