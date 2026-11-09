"""Predict an Alzheimer's MRI category from an image file."""

import argparse
from pathlib import Path

from PIL import Image

from code.alzheimer import load_alzheimer_model, predict_alzheimer_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a brain MRI image")
    image_path = parser.parse_args().image
    if not image_path.is_file():
        parser.error(f"Image file does not exist: {image_path}")

    model = load_alzheimer_model()
    with Image.open(image_path) as image:
        label, score = predict_alzheimer_image(image, model)

    print(f"Predicted category: {label}")
    print(f"Model score: {score:.1%}")
    print("This educational model does not provide a medical diagnosis.")


if __name__ == "__main__":
    main()
