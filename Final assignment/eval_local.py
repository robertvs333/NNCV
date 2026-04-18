"""
This script provides a local evaluation utility for your trained segmentation models
on the Cityscapes validation set. It calculates the Mean Intersection over Union (mIoU)
and per-class IoU scores.


### Arguments:
- `--data_dir`: Path to the Cityscapes dataset.
- `--model_path`: Path to your trained model's `.pt` checkpoint file.
- `--model_arch`: The architecture of your model ("deeplabv3plus" or "unet").
- `--Resnet_size`: (Optional, for DeepLabV3+ only) The size of the ResNet backbone (50 or 101).

Ensure that the `model.py` file is accessible from this script's location.
"""

import torch
import numpy as np
import argparse # Added missing import
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode
)

# Import your model
from ..model import DeepLabV3Plus, Model

# 1. Configuration
DATA_DIR = "./data/cityscapes" 
MODEL_PATH = "./checkpoints/UNet_CrossEntropy_64/best_model.pt"
NUM_CLASSES = 19

# 2. Correct Mapping for 19 Classes
# This creates a fast lookup array for mapping 0-33 IDs to 0-18 TrainIDs
mapping_array = np.zeros(256, dtype=np.uint8) + 255
for cls in Cityscapes.classes:
    if cls.train_id != 255 and cls.train_id != -1:
        mapping_array[cls.id] = cls.train_id

def argparser():
    parser = argparse.ArgumentParser(description="Evaluate models on Cityscapes Validation Set")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--model_arch", type=str, default="deeplabv3plus", choices=["deeplabv3plus", "unet"])
    parser.add_argument("--Resnet_size", type=int, default=50, choices=[50, 101], help="ResNet backbone size for DeepLabV3+")
    return parser.parse_args()

def fast_hist(target, prediction, num_classes):
    mask = (target >= 0) & (target < num_classes)
    # This math converts 2D coordinates into a 1D index for bincount
    hist = np.bincount(
        num_classes * target[mask].astype(int) + prediction[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def preprocess(img: Image.Image) -> torch.Tensor:
    # Use 512 if that's what you trained on, or 769 if you want higher resolution
    transform = Compose([
        ToImage(),
        Resize(size=512, interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    import torch.nn.functional as F
    # Get class indices
    pred_max = torch.argmax(pred, dim=1, keepdim=True)
    # Resize back to 1024x2048
    pred_resized = F.interpolate(pred_max.float(), size=original_shape, mode='nearest')
    return pred_resized.cpu().byte().squeeze().numpy()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    if args.model_arch == "deeplabv3plus":
        model = DeepLabV3Plus(Resnet_weights=False, n_classes=NUM_CLASSES, Resnet_size=args.Resnet_size)
    else:
        model = Model(n_classes=NUM_CLASSES)

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True) 
    model.load_state_dict(state_dict)
    model.eval().to(device)

    val_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic")
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for i in tqdm(range(len(val_dataset))):
        img, target = val_dataset[i]
        original_shape = (1024, 2048) 

        img_tensor = preprocess(img).to(device)
        with torch.no_grad():
            pred_logits = model(img_tensor)
        
        pred_mask = postprocess(pred_logits, original_shape)
        
        # Super fast mapping using the array we built
        target_mask = mapping_array[np.array(target)]

        hist += fast_hist(target_mask, pred_mask, NUM_CLASSES)

    # Intersection / Union
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iu)

    # Correct way to get the 19 class names in order
    train_id_to_name = {cls.train_id: cls.name for cls in Cityscapes.classes if cls.train_id != 255 and cls.train_id != -1}
    sorted_names = [train_id_to_name[i] for i in range(NUM_CLASSES)]

    print("\n" + "="*40)
    print(f"{args.model_arch.upper()} Evaluation Results")
    print(f"Mean IoU: {miou * 100:.2f}%")
    print("="*40)
    for name, iou in zip(sorted_names, iu):
        print(f"{name:>15}: {iou * 100:.2f}%")

if __name__ == "__main__":
    args = argparser()
    main(args)