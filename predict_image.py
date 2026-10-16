"""
Image Classification Inference Module for Single Vehicle Patches.
Accepts an image file or in-memory array, runs classifier inference,
reports class probabilities, inference latency, confidence scores,
and uncertainty warnings.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from classifier_models import load_checkpoint
from config import get_device
from dataset_utils import get_data_transforms
from utils import setup_logger

logger = setup_logger("predict_image")

EDUCATIONAL_NOTICE = (
    "NOTE: Softmax confidence reflects model output calibration across learned classes, "
    "not an absolute guarantee of correctness. This classifier is trained on isolated "
    "vehicle crops and cannot detect or localize multiple vehicles in uncropped scenes."
)


def predict_vehicle_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    checkpoint_path: Union[str, Path],
    confidence_threshold: float = 0.50,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    Predict vehicle class for a single vehicle crop.
    Args:
        image_input: File path, PIL Image, or RGB numpy array
        checkpoint_path: Path to trained .pth checkpoint
        confidence_threshold: Threshold below which prediction is marked uncertain
        device_name: 'auto', 'cuda', or 'cpu'
    Returns:
        Prediction dictionary with top-1 class, probabilities, latency, and warnings.
    """
    device = get_device(device_name)
    model, meta = load_checkpoint(checkpoint_path, device=device)
    class_names = meta.get("class_names", [])
    image_size = meta.get("image_size", 224)

    # Load and convert image to RGB PIL
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        pil_img = Image.open(img_path).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Preprocess
    transform = get_data_transforms(image_size=image_size, is_train=False, augment=False)
    tensor_img = transform(pil_img).unsqueeze(0).to(device)

    # Inference with precise timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_t = time.time()

    with torch.no_grad():
        logits = model(tensor_img)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.time() - start_t) * 1000.0

    top_idx = int(np.argmax(probabilities))
    top_class = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"
    top_confidence = float(probabilities[top_idx])

    is_uncertain = top_confidence < confidence_threshold

    prob_dict = {
        class_names[i] if i < len(class_names) else f"class_{i}": round(float(prob), 4)
        for i, prob in enumerate(probabilities)
    }

    # Sort probabilities descending
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    return {
        "predicted_class": top_class,
        "confidence": round(top_confidence, 4),
        "confidence_percentage": round(top_confidence * 100, 2),
        "is_uncertain": is_uncertain,
        "uncertainty_warning": (
            f"Prediction confidence ({top_confidence*100:.1f}%) is below configured threshold "
            f"({confidence_threshold*100:.1f}%)."
            if is_uncertain
            else None
        ),
        "class_probabilities": sorted_probs,
        "inference_latency_ms": round(latency_ms, 2),
        "architecture": meta.get("architecture", "unknown"),
        "image_size": image_size,
        "device": str(device),
        "educational_notice": EDUCATIONAL_NOTICE,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict vehicle class on single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to vehicle crop image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold for certainty")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    try:
        res = predict_vehicle_image(
            image_input=args.image,
            checkpoint_path=args.checkpoint,
            confidence_threshold=args.threshold,
            device_name=args.device,
        )
        print("\n" + "=" * 60)
        print("  Vehicle Classification Inference Result")
        print("=" * 60)
        print(f"Prediction: {res['predicted_class'].upper()}")
        print(f"Confidence: {res['confidence_percentage']}%")
        print(f"Latency:    {res['inference_latency_ms']} ms")
        print(f"Device:     {res['device']}")
        if res["is_uncertain"]:
            print(f"[!] WARNING: {res['uncertainty_warning']}")
        print("\nAll Class Probabilities:")
        for cls, prob in res["class_probabilities"].items():
            bar = "#" * int(prob * 30)
            print(f"  {cls:<15} : {prob*100:5.1f}%  |{bar}")
        print("\n" + res["educational_notice"])
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
