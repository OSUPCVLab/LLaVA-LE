"""
Lunar Surface Caption Generator using GPT-4 Vision

This module provides classes and functions to generate scientific captions for lunar surface imagery
using OpenAI's GPT Vision API. It processes multimodal datasets containing panchromatic, gravity,
and slope data.
"""

import os
import base64
from pathlib import Path
from PIL import Image
from openai import OpenAI
import random


STYLE_TEMPLATES = [
    "Begin by describing spatial layout from any part of the frame.",
    "Start by identifying the dominant visible feature.",
    "Start with a geological process suggested by the surface form.",
    "Begin by describing contrasts between different zones, if any.",
    "Start by highlighting the most prominent landmark.",
    "Start by describing spatial relationships between major features.",
    "Begin with visible structural or textural patterns.",
    "Start by describing elevation or slopes.",
    "Begin with relative age clues based on surface form.",
    "Start by describing a boundary zone, if any.",
    "Begin with a multi-feature summary.",
    "Start with possible subsurface density suggested by the surface form.",
    "Begin with a sentence linking surface form to possible subsurface tendencies.",
    "Start with a directional description (west-to-east, north-to-south, etc.).",
    "Begin by describing noticeable geometry and shapes.",
    "Start with a statement about arrangement of features.",
    "Begin by describing crater-related structures.",
    "Start by describing the plains or broad flat areas."
]

BASE_PROMPT = """
You are a lunar surface scientist with expertise in impact processes, volcanism, and crustal structure. You receive three co-registered views of the same location: an optical image, a Bouguer gravity anomaly map, and a terrain slope map. You may use all three internally for your reasoning, but the caption must read as if it relies only on the optical view. Never mention or imply gravity, slope, elevation, measurements, data layers, or sensing modalities.
Write a 3–4 sentence scientific caption (≤80 words). Provide a clear description of the surface: texture, crater forms, relative locations of major features, and broader patterns across the frame. Note crater distribution and any visible lineations, boundaries, or smooth/rough domains, but avoid over-interpreting small or ambiguous details.
Include one or two simple accurate geological inferences that an expert might plausibly draw from morphology alone. These inferences must subtly reflect subsurface tendencies suggested by gravity (e.g., denser or less dense crust, thickness of fill, structural stability) but written so that they appear based only on surface texture. Frame such points with careful language such as “may reflect,” “could indicate,” “is consistent with,” or “likely relates to.”
Avoid confident or overly specific interpretations. Never mention colors, maps, datasets, or modalities. The caption must sound like a natural, concise expert interpretation based solely on an optical image.

"### Caption Requirements:"
- Write in clear, academic American English (no poetic or subjective tone).
- Do NOT mention the words 'panchromatic', 'gravity', or 'slope'. Integrate their implications implicitly.
- Avoid color words (e.g., red, blue, yellow) or references to map/layer/modality.
- The caption must sound like an imaging report written by a lunar geoscientist.
- Each caption must include one cautious inference about subsurface density written as if it is inferred from surface morphology only.
- Each caption must include one observation about apparent slope conditions phrased as if inferred from surface texture only. 
- Do NOT mention or hint at the existence of other data layers (e.g., gravity, slope, maps, subsurface values, or measurements).

"### Interpretation Rules (for your reasoning only):"
"• Bouguer Gravity (−300 to +600 mGal)\n"
"  - Deep purple–blue (−300 to −100): strong negative anomaly, means low-density subsurface. Suggests thick regolith, voids, or basin fill beneath more subdued or infilled surfaces. \n"
"  - Green–yellow (−50 to 0): near-neutral. Suggests average crustal density. \n"
"  - Orange–red (0 to +400): moderate positive anomaly, means high density subsurface regions. Can point to basaltic fill, crustal thinning, or uplifted deeper material, especially if they coincide with broad, smooth plains or basin centers. \n"
"  - Brown–dark brown (+400 to +600): strong positive anomaly. Suggests high density subsurface material,  mascon, mantle uplift, or dense fill.\n"
"• Terrain Slope (0°–40°)\n"
"  - Light pink (~0°): flat plains or infilled basins \n"
"  - Blue (~5–20°): moderate slopes on ridges, crater walls, or degraded terrain.\n"
"  - Yellowish-brown (~20–40°): steep rims, scarps, or structural boundaries.\n"

### Geologic interpretation guidance (for your reasoning only; never include these phrases directly)
Strong gravity anomaly + low slope suggests likely dense subsurface fill under smooth surface
Strong gravity anomaly + high slope suggests possible basement uplift/exposure of dense material
Low gravity amomaly + low slope suggests thick low-density basin sediments/regolith
Low gravity anomaly + high slope → possibly low-density crustal block or heavily fractured/porous terrain.
• Smooth, dark plains suggests dense basaltic mare.
• Bright, rugged terrain can suggest anorthositic highlands.
• Steep or rough linear ridges can suggest tectonic deformation or basin rims.
• When you discuss subsurface or tectonic tendencies in the caption, always frame them as possibilities or likelihoods inferred from surface form.
• If an area is Blue/Purple, you are FORBIDDEN from describing it as "dense," "mascon," "uplift," and "mantle." 
• If an area is Red/Orange/Brown, you are FORBIDDEN from describing it as "porous", "low-density", "voids" or "thick regolith."
•**Note:** Gravity data always trumps visual texture when determining subsurface density.


###Description Patterns

Each caption must choose freely from the following pattern sets and apply them in different combinations:

Surface-description order:
- Start with a dominant feature, or with texture trends, or with crater distribution, or with directional changes, or with the largest landform, or with subtle textures that scale outward, or with a smooth–rugged comparison, or with fine textures that expand to regional context, or with a central-to-peripheral hierarchy, or with brightness or texture contrasts before structures.

Spatial language:
- Prioritize central features, or contrast margins, or describe diagonally, or move from near to far, or express lineations as trending/oriented/aligned, or use relational framing such as “set within” or “bounded by”, or treat crater density as a gradient, or describe boundaries as sharp/diffuse/transitional/gradational.

Surface–subsurface reasoning:
- Use statements linking surface form to material contrasts, crustal strength, subsurface coherence, buried structures, density contrasts at depth, stable underlying material, infill, or resistant units if any.

Slope inference:
- Use wording implying slopes, topography, crater segments, inclines, relief, transitions, varying steepness, or flat plains if any.

Limit including descriptions about volcanism, tectonics, crustal strength, resurfacing, basins to cases where the surface form provides direct and clear evidence. 
Each caption should combine a selection from these sets but not reproduce sentences verbatim.

Use these cues flexibly and internally to vary sentence openings, ordering, phrasing, and interpretive style across captions. Never quote these patterns directly.

"Now write one caption for this triplet:"
"<image 1: gravity>\n"
"<image 2: panchro>\n"
"<image 3: slope>"
"""


class LunarCaptionGenerator:


    def __init__(self, api_key: str):
        """
        Initialize the caption generator.

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)

        self.system_prompt = (
            "You are a lunar surface scientist with deep expertise in impact processes, volcanic resurfacing, structural geology, and crustal evolution. You receive three co-registered observations of the same location: (1) an optical image, (2) a Bouguer gravity map (degree 6 to 660), and (3) a terrain slope map. You may use all three sources internally to guide your scientific reasoning, but everything you write must read as if it is based only on the optical image. Never mention or imply the presence of non-optical data. "
            "You analyze co-registered lunar datasets from orbital sensors and write professional imaging captions for scientific figures. Your tone is concise, factual, and academic. Avoid poetic or subjective language. "
            "Your captions should combine detailed spatial description of visible features with expert-level geological interpretation that an experienced lunar researcher could infer from surface morphology alone. This includes subsurface tendencies, crustal density contrasts, mechanical strength differences, resurfacing signatures, impact modification state, tectonic structure, and relative age."
            "Never refer to the data sources, maps, datasets, measurements, colors, image source or sensing modalities used to infer your interpretation."
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
            #style hint
            style_hint = random.choice(STYLE_TEMPLATES)
            
            # Encode all images as base64
            encoded_pancro = self.encode_pil_image(pancro_image)
            encoded_gravity = self.encode_pil_image(gravity_image)
            encoded_slope = self.encode_pil_image(slope_image)

            # Prepare message content
            message_content = [
                {
                    "type": "text", 
                    "text": f"{BASE_PROMPT.strip()}\n\nSTYLE INSTRUCTION:\n- {style_hint}"
                },

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
                max_completion_tokens=300,
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
            #style hint
            style_hint = random.choice(STYLE_TEMPLATES)
            
            # Encode all images
            encoded_images = [self.encode_image_from_path(path) for path in image_paths]

            # Prepare message content
            message_content = [
                {
                    "type": "text", 
                    "text": f"{BASE_PROMPT.strip()}\n\nSTYLE INSTRUCTION:\n- {style_hint}"
                },
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
                max_completion_tokens=300,
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


# --- Add this code block at the very end of gpt_g.py ---

import argparse
# You will also need to add 'import json' at the top if you want to save the output

if __name__ == "__main__":
    # 1. Define and parse arguments
    parser = argparse.ArgumentParser(description="Generate lunar surface captions using GPT-4 Vision.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing lunar image triplets.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path for saving captions.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name to use (e.g., gpt-4o, gpt-4-vision).")
    parser.add_argument("--max_samples", type=int, default=1, help="Maximum number of samples to process.")
    parser.add_argument("--api_key", type=str, required=True, help="Your OpenAI API Key.")

    args = parser.parse_args()

    # 2. Check for the API Key and initialize the generator
    generator = LunarCaptionGenerator(api_key=args.api_key)

    # 3. Validate the model name
    if not generator.validate_model(args.model):
        print("Exiting due to invalid model.")
        # Optionally exit or switch to default model
        # args.model = "gpt-4o"
        # print(f"Switching to default model: {args.model}")

    # 4. Implement the logic to find image triplets in args.input
    #    and call generator.generate_caption for each one.
    #    This part is complex as it requires file system traversal.

    print(f"Generator initialized. Ready to process {args.max_samples} samples from {args.input} using model {args.model}.")