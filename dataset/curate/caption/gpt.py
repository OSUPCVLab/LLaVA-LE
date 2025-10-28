"""
Lunar Surface Caption Generator using GPT-4 Vision

This module provides classes and functions to generate scientific captions for lunar surface imagery
using OpenAI's GPT-4 Vision API. It processes multimodal datasets containing panchromatic, gravity,
and slope data.
"""

import os
import base64
from pathlib import Path
from PIL import Image
from openai import OpenAI


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

    def __init__(self, api_key: str):
        """
        Initialize the caption generator.

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)

        self.system_prompt = (
            "You are a lunar surface scientist with expertise in remote sensing and planetary geology. "
            "You analyze co-registered lunar datasets from orbital sensors and write professional imaging captions "
            "for scientific figures. Your tone is concise, factual, and academic. Avoid poetic or subjective language. "
            "Do *not* mention the maps gravity, or slope directly. Instead, integrate their implications implicitly. "
            "Describe the *surface morphology*, **apparent structure**, and **inferred subsurface properties**."
            "Do *not* mention the words 'positive', 'negative', 'anomaly'"
        )

    def get_available_models(self) -> list:
        """
        Get list of all available OpenAI models.

        Returns:
            List of available model names
        """
        try:
            models = self.client.models.list()
            available_models = []
            for m in models.data:
                available_models.append(m.id)

            # Sort and return unique models
            return sorted(list(set(available_models)))
        except Exception as e:
            raise ValueError(f"Warning: Could not fetch available models: {e}")

    def validate_model(self, model: str) -> bool:
        """
        Validate if the specified model is available.

        Args:
            model: Model name to validate

        Returns:
            True if model is valid, False otherwise
        """
        available_models = self.get_available_models()
        if model in available_models:
            return True

        print(f"Error: Model '{model}' not found in available models.")

        # Show vision-capable models as recommendations
        vision_models = [
            m
            for m in available_models
            if any(pattern in m for pattern in ["gpt-4", "gpt-4o", "gpt-4-vision"])
        ]
        if vision_models:
            print(
                f"Some available models: {', '.join(vision_models[:5])}"
            )  # Show first 5

        print(f"Total available models: {len(available_models)}")
        print("Use --list-models to see all available models.")
        return False

    def encode_pil_image(self, pil_image: Image.Image) -> str:
        """
        Encode a PIL Image as base64 for OpenAI API.

        Args:
            pil_image: PIL Image to encode

        Returns:
            Base64 encoded image string
        """
        from io import BytesIO

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

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

    def generate_caption_from_images(
        self,
        pancro_image: Image.Image,
        gravity_image: Image.Image,
        slope_image: Image.Image,
        model: str = "gpt-4o",
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a scientific caption from PIL Images.

        Args:
            pancro_image: Panchromatic PIL image
            gravity_image: Gravity anomaly PIL image
            slope_image: Slope map PIL image
            model: OpenAI model name to use for caption generation
            temperature: Temperature for text generation (0.0-2.0)

        Returns:
            Generated caption as string, or None if failed
        """
        try:
            # Encode all images as base64
            encoded_pancro = self.encode_pil_image(pancro_image)
            encoded_gravity = self.encode_pil_image(gravity_image)
            encoded_slope = self.encode_pil_image(slope_image)

            # Prepare message content
            message_content = [
                {"type": "text", "text": BASE_PROMPT.strip()},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_gravity}"},
                },  # Gravity (image 1)
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_pancro}"},
                },  # Panchro (image 2)
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_slope}"},
                },  # Slope (image 3)
            ]

            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message_content},
                ],
                temperature=temperature,
                max_tokens=300,
            )

            caption = response.choices[0].message.content.strip()
            return caption

        except Exception as e:
            print(f"Error generating caption: {e}")
            return None

    def generate_caption(
        self,
        image_paths: list,
        identifier: str = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
    ) -> dict:
        """
        Generate a scientific caption for a set of lunar images (legacy method for file paths).

        Args:
            image_paths: List of paths to [panchromatic, gravity, slope] images (will be reordered to match prompt: gravity, panchromatic, slope)
            identifier: Optional custom identifier for the dataset
            model: OpenAI model name to use for caption generation
            temperature: Temperature for text generation (0.0-2.0)

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
                    "image_url": {"url": f"data:image/png;base64,{encoded_images[1]}"},
                },  # Gravity (image 1)
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_images[0]}"},
                },  # Panchro (image 2)
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_images[2]}"},
                },  # Slope (image 3)
            ]

            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message_content},
                ],
                temperature=temperature,
                max_tokens=300,
            )

            caption = response.choices[0].message.content.strip()

            # Prepare result dictionary
            result = {
                "identifier": identifier,
                "caption": caption,
                "metadata": {
                    "temperature": temperature,
                    "word_count": len(caption.split()),
                },
            }

            return result

        except Exception as e:
            print(f"Error generating caption: {e}")
            return None

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
