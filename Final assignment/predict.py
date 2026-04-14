from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)

from model import DeepLabV3Plus # Ensure this matches your local file

IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

def preprocess(img: Image.Image) -> torch.Tensor:
    # 1. Resize to the target wide canvas
    img_resized = F.resize(img, size=(769, 1538), interpolation=InterpolationMode.BILINEAR)
    
    # 2. Extract Left and Right halves
    # crop(img, top, left, height, width)
    left_half = F.crop(img_resized, 0, 0, 769, 769)
    right_half = F.crop(img_resized, 0, 769, 769, 769)
    
    # 3. Combine into a batch [2, 3, 769, 769]
    batch = torch.stack([ToImage()(left_half), ToImage()(right_half)])
    
    # 4. Final normalization and dtype conversion
    transform = Compose([
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform(batch)


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    """
    pred: Tensor of shape [2, C, 769, 769] (Batch of 2: Left and Right)
    original_shape: (Height, Width)
    """
    # 1. Get class predictions for both halves
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1)  # Shape: [2, 769, 769]

    # 2. Stitch the two 769x769 halves horizontally back to 1538x769
    # Dim 1 is width here because batch dim was removed by argmax
    stitched_mask = torch.cat([pred_max[0], pred_max[1]], dim=1) 
    
    # 3. Add batch and channel dims for Resize: [1, 1, 769, 1538]
    stitched_mask = stitched_mask.unsqueeze(0).unsqueeze(0)

    # 4. Resize back to original image dimensions
    prediction = F.resize(
        stitched_mask, 
        size=original_shape, 
        interpolation=InterpolationMode.NEAREST
    )

    prediction_numpy = prediction.cpu().detach().numpy()
    return prediction_numpy.squeeze().astype(np.uint8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = DeepLabV3Plus(Resnet_weights=False)
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
            img_batch = preprocess(img).to(device)
            
            # Model processes both halves in one forward pass
            pred = model(img_batch) # Output: [2, C, 769, 769]

            # Postprocess handles stitching and final resize
            seg_pred = postprocess(pred, original_shape)
            
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(seg_pred).save(out_path)


if __name__ == "__main__":
    main()