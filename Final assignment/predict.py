"""
This script is designed to perform inference using a trained model within a Docker container
for the Cityscapes Challenge submission. It expects specific input and output directory
structures and a model weight file at a predefined path.

### How to use this script (within Docker):
This script is the entrypoint for the Docker container. It will be executed automatically
when the Docker container is run.

- **Input:** It reads `.png` images from the `IMAGE_DIR` ("/data").
- **Model:** It loads the model weights from `MODEL_PATH` ("/app/model.pt").
- **Output:** It saves the predicted segmentation masks as `.png` files to the `OUTPUT_DIR` ("/output").

### Configuration:
- `ARCH`: Set this to "unet" or "deeplabv3plus" based on your trained model.
- `RESNET_SIZE`: If using "deeplabv3plus", set the ResNet backbone size (50 or 101).

Before building the Docker image, ensure your best trained model checkpoint is copied to
`Final assignment/model.pt` in your repository root.
"""

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)

from model import DeepLabV3Plus, Model # Ensure this matches your local file

IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

def preprocess(img: Image.Image, architecture="deeplabv3plus") -> torch.Tensor:
    # Set size based on architecture: 513 for DeepLab, 512 for U-Net
    target_size = 513 if architecture == "deeplabv3plus" else 512
    
    transform = Compose([
        ToImage(),
        # Passing a single int resizes the SHORTER edge and maintains aspect ratio
        Resize(size=target_size, interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Returns [1, 3, H, W]
    return transform(img).unsqueeze(0)


def postprocess(logits: torch.Tensor, original_shape: tuple) -> np.ndarray:
    """
    logits: Tensor of shape [1, 19, H, W]
    original_shape: (height, width) e.g., (1024, 2048)
    """
    # 1. Get class indices: [1, 19, H, W] -> [1, 1, H, W]
    pred_max = torch.argmax(logits, dim=1, keepdim=True)

    # 2. Resize back to original image dimensions (1024x2048)
    # We use nearest neighbor to ensure we don't create "fake" class IDs
    pred_resized = F.interpolate(
        pred_max.float(), 
        size=original_shape, 
        mode='nearest'
    )
    
    prediction_numpy = pred_resized.cpu().detach().numpy()
    return prediction_numpy.squeeze().astype(np.uint8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ADJUST THESE based on your experiment
    ARCH = "unet" 
    RESNET_SIZE = 101
    # Load model
    if ARCH == "deeplabv3plus":
        model = DeepLabV3Plus(Resnet_weights=False, Resnet_size=RESNET_SIZE)
    else:
        model = Model()
    state_dict = torch.load(
        MODEL_PATH, 
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = (img.height, img.width)

            # Preprocess returns a batch of 2 (Left/Right)
            img_batch = preprocess(img, architecture=ARCH).to(device)
            
            # Model processes both halves in one forward pass
            pred = model(img_batch) # Output: [2, C, 769, 769]

            # Postprocess handles stitching and final resize
            seg_pred = postprocess(pred, original_shape)
            
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(seg_pred).save(out_path)


if __name__ == "__main__":
    main()