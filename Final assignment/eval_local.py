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

import os
import torch
import numpy as np
import pandas as pd
import argparse
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

# 2. Super-Category Mapping (TrainIDs)
SUPER_CATEGORIES = {
    "Flat": [0, 1],             # road, sidewalk
    "Const.": [2, 3, 4],        # building, wall, fence
    "Obj.": [5, 6, 7],          # pole, traffic light, traffic sign
    "Nat.": [8, 9],             # vegetation, terrain
    "Sky": [10],                # sky
    "Hum.": [11, 12],           # person, rider
    "Veh.": [13, 14, 15, 16, 17, 18] # car, truck, bus, train, motorcycle, bicycle
}

# 3. Correct Mapping for 19 Classes & Name Mapping for Printing
mapping_array = np.zeros(256, dtype=np.uint8) + 255
trainid_to_name = {}  # Added to grab human-readable names

for cls in Cityscapes.classes:
    if cls.train_id != 255 and cls.train_id != -1:
        mapping_array[cls.id] = cls.train_id
        trainid_to_name[cls.train_id] = cls.name

def argparser():
    parser = argparse.ArgumentParser(description="Evaluate models on Cityscapes Validation Set")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--model_arch", type=str, default="deeplabv3plus", choices=["deeplabv3plus", "unet"])
    parser.add_argument("--Resnet_size", type=int, default=50, choices=[50, 101], help="ResNet backbone size for DeepLabV3+")
    parser.add_argument("--freeze_backbone", action="store_true", help="Whether to freeze the backbone during evaluation")
    return parser.parse_args()

def fast_hist(target, prediction, num_classes):
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + prediction[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=(512, 1024), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    import torch.nn.functional as F
    pred_max = torch.argmax(pred, dim=1, keepdim=True)
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

    for i in tqdm(range(len(val_dataset)), desc=f"Evaluating {args.model_arch}"):
        img, target = val_dataset[i]
        original_shape = (1024, 2048) 

        img_tensor = preprocess(img).to(device)
        with torch.no_grad():
            pred_logits = model(img_tensor)
        
        pred_mask = postprocess(pred_logits, original_shape)
        target_mask = mapping_array[np.array(target)]

        hist += fast_hist(target_mask, pred_mask, NUM_CLASSES)

    # Calculate global IoU per class properly
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    
    # Avoid division by zero warnings by masking
    valid_classes = union > 0
    iu = np.full(NUM_CLASSES, np.nan)
    iu[valid_classes] = intersection[valid_classes] / union[valid_classes]
    
    miou = np.nanmean(iu) * 100  # Multiplied by 100 for percentage

    # Calculate Super-Category averages PROPERLY (Intersection / Union)
    grouped_results = {}
    for cat_name, class_ids in SUPER_CATEGORIES.items():
        cat_tp = np.sum(hist[np.ix_(class_ids, class_ids)])
        cat_gt = np.sum(hist[class_ids, :])
        cat_pred = np.sum(hist[:, class_ids])
        cat_union = cat_gt + cat_pred - cat_tp
        
        if cat_union > 0:
            cat_iou = (cat_tp / cat_union) * 100 
        else:
            cat_iou = np.nan
            
        grouped_results[cat_name] = cat_iou
    
    # Add final mIoU
    grouped_results["mIoU"] = miou

    # --- PRINT TO TERMINAL ---
    print("\n" + "="*50)
    print(f"RESULTS: {args.model_arch.upper()} | R{args.Resnet_size if args.model_arch == 'deeplabv3plus' else 'N/A'} | {'CE' if 'CrossEntropy' in args.model_path else 'Focal'}")
    
    # Print Individual Classes First
    print("="*50)
    print(f"{'PER-CLASS IOU':^50}")
    print("="*50)
    for train_id in range(NUM_CLASSES):
        class_name = trainid_to_name.get(train_id, f"Class_{train_id}")
        class_score = iu[train_id] * 100
        print(f"{class_name:>15}: {class_score:.2f}%")

    # Print Category Summaries Next
    print("-" * 50)
    print(f"{'CATEGORY IOU':^50}")
    print("-" * 50)
    for cat, score in grouped_results.items():
        print(f"{cat:>15}: {score:.2f}%")  
    print("="*50)

    # Export to CSV Table
    # Note: I am keeping this to just export the categories and mIoU so it doesn't break your existing Overleaf table script.
    df = pd.DataFrame([grouped_results]).round(2)
    
    df.insert(0, "Model", args.model_arch)
    df.insert(1, "Loss", 'CrossEntropy' if 'CrossEntropy' in args.model_path else 'Focal')
    
    export_name = args.model_arch + "_" + ('CE' if 'CrossEntropy' in args.model_path else 'Focal') + ".csv"
    file_exists = os.path.isfile(export_name)
    
    df.to_csv(export_name, mode='a', index=False, header=not file_exists)
    
    print(f"\nResults cleanly appended to {export_name}")

if __name__ == "__main__":
    args = argparser()
    main(args)