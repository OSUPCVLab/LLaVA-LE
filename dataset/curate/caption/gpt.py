import os
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from PIL import Image
from openai import OpenAI
from lumina_dataset import LUMINADataset, create_split_datasets


BASE_PROMPT = """
"You receive three co-registered images of the same lunar location:\n"
"1. Panchromatic (surface reflectance and morphology)\n"
"2. Bouguer gravity anomaly (degree 6–660, range −300 to +600 mGal)\n"
"3. Terrain slope map (range 0° to 40°)\n\n"
"Each image represents the same area at the same spatial scale.\n\n"
"Your task is to write a concise, scientific figure caption describing what an expert would infer "
"from all three images together, focusing on geological interpretation rather than raw color information.\n\n"
"### Caption Requirements:\n"
"- Write in clear, academic American English (no poetic or subjective tone).\n"
"- Use 3–5 sentences (≤110 words).\n"
"- Do NOT mention the words 'panchromatic', 'gravity', or 'slope'. Integrate their implications implicitly.\n"
"- Describe surface morphology, apparent structure, and inferred subsurface properties (density, stability, composition, or age).\n"
"- Avoid color words (e.g., red, blue, yellow) or references to map/layer/modality.\n"
"- The caption must sound like an imaging report written by a lunar geoscientist.\n\n"
"### Interpretation Rules (for your reasoning only, not to include in the caption):\n"
"• Bouguer Gravity (−300 to +600 mGal)\n"
"  - Deep purple–blue (−300 to −100): strong negative anomaly → low-density crust, basin fill, or voids.\n"
"  - Green–yellow (−50 to 0): near-neutral → average crustal density.\n"
"  - Orange–red (0 to +400): moderate positive anomaly → denser subsurface, basaltic flows, or crustal thinning.\n"
"  - Brown–dark brown (+400 to +600): strong positive anomaly → mascon, mantle uplift, or dense fill.\n"
"• Terrain Slope (0°–40°)\n"
"  - Light pink (~0°): flat plains or volcanic infill.\n"
"  - Blue (~5–20°): moderate slopes on ridges, crater walls, or degraded terrain.\n"
"  - Yellowish-brown (~20–40°): steep rims, scarps, or structural boundaries.\n"
"• Smooth, dark plains → dense basaltic mare (young, high gravity, low slope).\n"
"• Bright, rugged terrain → anorthositic highlands (low gravity, high slope).\n"
"• Steep or rough linear ridges → tectonic deformation or basin rims.\n"
"• Circular depressions → impact craters; subdued rims indicate age or relaxation.\n"
"• Positive anomalies → dense basalt infill or uplifted mantle.\n"
"• Negative anomalies → low-density crust or thick regolith.\n"
"• High slopes (~40°) → steep crater rims or scarps; low slopes → plains or infilled basins.\n\n"
"Now write one caption for this triplet:\n"
"<image 1: gravity>\n"
"<image 2: panchro>\n"
"<image 3: slope>"
"""


class LunarCaptionGenerator:
    """
    A class to generate scientific captions for lunar surface imagery using OpenAI's GPT-4 Vision.
    """

    def __init__(self, api_key: str, output_dir: str = "outputs"):
        """
        Initialize the caption generator.

        Args:
            api_key: OpenAI API key
            output_dir: Directory to save output JSON files
        """
        self.client = OpenAI(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.system_prompt = (
            "You are a lunar surface scientist with expertise in remote sensing and planetary geology. "
            "You analyze co-registered lunar datasets from orbital sensors and write professional imaging captions "
            "for scientific figures. Your tone is concise, factual, and academic. Avoid poetic or subjective language. "
            "Do *not* mention the maps gravity, or slope directly. Instead, integrate their implications implicitly. "
            "Describe the *surface morphology*, **apparent structure**, and **inferred subsurface properties**."
            "Do *not* mention the words 'positive', 'negative', 'anomaly'"
        )

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a PyTorch tensor to PIL Image.

        Args:
            tensor: Input tensor of shape (C, H, W) with values in [0, 1] or [0, 255]

        Returns:
            PIL Image
        """
        if tensor.dim() == 3:
            # Convert from (C, H, W) to (H, W, C)
            tensor = tensor.permute(1, 2, 0)

        # Normalize to [0, 255] if needed
        if tensor.max() <= 1.0:
            tensor = tensor * 255

        # Convert to numpy array and ensure uint8
        array = tensor.cpu().numpy().astype(np.uint8)

        # Handle grayscale and RGB
        if array.shape[-1] == 1:
            array = array.squeeze(-1)
            return Image.fromarray(array, mode="L")
        else:
            return Image.fromarray(array, mode="RGB")

    def encode_tensor_image(self, tensor: torch.Tensor) -> str:
        """
        Encode a PyTorch tensor as base64 image for OpenAI API.

        Args:
            tensor: Input tensor representing an image

        Returns:
            Base64 encoded image string
        """
        # Convert tensor to PIL Image
        pil_image = self.tensor_to_pil(tensor)

        # Save to bytes buffer
        from io import BytesIO

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode to base64
        import base64

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def encode_image_from_path(self, image_path: str) -> str:
        """
        Encode image file to base64 format required by OpenAI API.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_caption_from_tensors(
        self,
        pancro_tensor: torch.Tensor,
        gravity_tensor: torch.Tensor,
        slope_tensor: torch.Tensor,
        identifier: str,
    ) -> dict:
        """
        Generate a scientific caption from PyTorch tensors.

        Args:
            pancro_tensor: Panchromatic image tensor
            gravity_tensor: Gravity anomaly tensor
            slope_tensor: Slope map tensor
            identifier: Unique identifier for the dataset

        Returns:
            Dictionary containing the generated caption and metadata
        """
        try:
            # Encode all images as base64
            encoded_pancro = self.encode_tensor_image(pancro_tensor)
            encoded_gravity = self.encode_tensor_image(gravity_tensor)
            encoded_slope = self.encode_tensor_image(slope_tensor)

            # Prepare message content
            message_content = [
                {"type": "text", "text": BASE_PROMPT.strip()},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_pancro}"},
                },  # Panchro
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_gravity}"},
                },  # Gravity
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_slope}"},
                },  # Slope
            ]

            # Make API call
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message_content},
                ],
                temperature=0.3,
                max_tokens=300,
            )

            caption = response.choices[0].message.content.strip()

            # Prepare result dictionary
            result = {
                "identifier": identifier,
                "caption": caption,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": "gpt-4o",
                    "temperature": 0.3,
                    "word_count": len(caption.split()),
                },
            }

            return result

        except Exception as e:
            print(f"Error generating caption for {identifier}: {e}")
            return None

    def generate_caption(self, image_paths: list, identifier: str = None) -> dict:
        """
        Generate a scientific caption for a set of lunar images (legacy method for file paths).

        Args:
            image_paths: List of paths to [panchromatic, gravity, slope] images
            identifier: Optional custom identifier for the dataset

        Returns:
            Dictionary containing the generated caption and metadata
        """
        if not self.validate_image_paths(image_paths):
            return None

        # Extract identifier if not provided
        if identifier is None:
            identifier = self.extract_identifier(image_paths)

        try:
            # Encode all images
            encoded_images = [self.encode_image_from_path(path) for path in image_paths]

            # Prepare message content
            message_content = [
                {"type": "text", "text": BASE_PROMPT.strip()},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_images[0]}"},
                },  # Panchro
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_images[1]}"},
                },  # Gravity
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_images[2]}"},
                },  # Slope
            ]

            # Make API call
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message_content},
                ],
                temperature=0.3,
                max_tokens=300,
            )

            caption = response.choices[0].message.content.strip()

            # Prepare result dictionary
            result = {
                "identifier": identifier,
                "caption": caption,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": "gpt-4o",
                    "temperature": 0.3,
                    "word_count": len(caption.split()),
                },
            }

            return result

        except Exception as e:
            print(f"Error generating caption: {e}")
            return None

    def process_lumina_dataset(
        self,
        dataset: "LUMINADataset",
        max_samples: Optional[int] = None,
        output_filename: str = None,
    ) -> str:
        """
        Process an entire LUMINADataset and generate captions for all triplets.

        Args:
            dataset: LUMINADataset instance
            max_samples: Maximum number of samples to process (None for all)
            output_filename: Custom output filename (without extension)

        Returns:
            Path to the saved JSON file
        """
        total_samples = len(dataset)
        num_samples = min(max_samples, total_samples) if max_samples else total_samples

        print(
            f"Processing {num_samples} samples from LUMINADataset (total: {total_samples})"
        )

        # Collect all results
        all_results = {
            "dataset_info": {
                "total_samples": total_samples,
                "processed_samples": num_samples,
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4o",
            },
            "captions": [],
        }

        # Process each sample
        successful_count = 0
        for i in range(num_samples):
            try:
                print(f"Processing sample {i+1}/{num_samples}...", end=" ")

                # Get sample from dataset
                sample = dataset[i]

                # Extract tensors
                pancro_tensor = sample["pancro"]
                gravity_tensor = sample["gravity"]
                slope_tensor = sample["slope"]

                # Create identifier from dataset index and tile info
                # Get the actual tile and row_col info from the dataset
                tile_name, row_col = dataset.common_index[i]
                identifier = f"{tile_name}_{row_col}"

                # Generate caption
                result = self.generate_caption_from_tensors(
                    pancro_tensor, gravity_tensor, slope_tensor, identifier
                )

                if result:
                    all_results["captions"].append(result)
                    successful_count += 1
                    print(f"✓ Caption generated for {identifier}")
                else:
                    print(f"✗ Failed to generate caption for {identifier}")

            except Exception as e:
                print(f"✗ Error processing sample {i}: {e}")
                continue

        # Save all results to a single JSON file
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"lumina_captions_{timestamp}"

        filepath = self.output_dir / f"{output_filename}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Successfully generated: {successful_count}/{num_samples} captions")
        print(f"Results saved to: {filepath}")

        return str(filepath)

    def validate_image_paths(self, image_paths: list) -> bool:
        """
        Validate that all image files exist.

        Args:
            image_paths: List of image file paths

        Returns:
            True if all files exist, False otherwise
        """
        missing_files = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                missing_files.append(img_path)

        if missing_files:
            print("Error: The following image files are missing:")
            for file in missing_files:
                print(f"  - {file}")
            return False

        print("All image files found successfully.")
        return True

    def extract_identifier(self, image_paths: list) -> str:
        """
        Extract a unique identifier from image file paths.

        Args:
            image_paths: List of image file paths

        Returns:
            Unique identifier string
        """
        # Extract common part from filenames (e.g., "N19.500_W01.000" from multiple files)
        first_path = Path(image_paths[0])
        filename = first_path.stem

        # Remove the modality suffix (e.g., "_panchro", "_gravty", "_slope")
        parts = filename.split("_")
        if len(parts) > 1:
            # Keep everything except the last part (which should be the modality)
            identifier = "_".join(parts[:-1])
        else:
            identifier = filename

        return identifier

    def save_result(self, result: dict, filename: str = None) -> str:
        """
        Save the caption result to a JSON file.

        Args:
            result: Dictionary containing caption and metadata
            filename: Optional custom filename (without extension)

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"caption_{result['identifier']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        filepath = self.output_dir / f"{filename}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Caption saved to: {filepath}")
        return str(filepath)


def main():
    """Main function to run the caption generation."""
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY")

    # Initialize caption generator
    generator = LunarCaptionGenerator(api_key=api_key, output_dir="outputs")

    # Option 1: Process LUMINADataset (recommended)
    if LUMINADataset is not None:
        print("Using LUMINADataset for processing...")

        # Initialize the dataset (adjust paths as needed)
        root_dir = "path/to/your/dataset"  # Change this to your dataset path
        splits_file = "path/to/splits.json"  # Change this to your splits file

        # You can process a specific split or the entire dataset
        try:
            # Create dataset instance
            dataset = LUMINADataset(
                root_dir=root_dir,
                # split="train",  # Uncomment to process only training split
                # splits_file=splits_file,  # Uncomment if using splits
            )

            # Process all samples (or limit with max_samples parameter)
            result_file = generator.process_lumina_dataset(
                dataset,
                max_samples=10,  # Remove or set to None to process all samples
                output_filename="lumina_captions_batch",
            )

            print(f"All captions saved to: {result_file}")

        except Exception as e:
            print(f"Error processing LUMINADataset: {e}")
            print("Falling back to individual file processing...")
            process_individual_files(generator)
    else:
        print("LUMINADataset not available, using individual file processing...")
        process_individual_files(generator)


if __name__ == "__main__":
    main()
