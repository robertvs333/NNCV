import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode
)

# Import your model
from model import DeepLabV3Plus

# 1. Configuration
DATA_DIR = "./data/cityscapes" # Change to your local Cityscapes path
MODEL_PATH = "model2.pt" # Change to your 0.24 loss model
NUM_CLASSES = 19
IGNORE_INDEX = 255

# 2. Cityscapes ID to Train ID Mapping (For the Ground Truth)
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

def convert_target_to_trainid(target_img: Image.Image) -> np.ndarray:
    target_np = np.array(target_img)
    # Vectorized mapping for speed
    mapped_target = np.vectorize(id_to_trainid.get)(target_np)
    return mapped_target

# 3. Fast Confusion Matrix (Industry standard for IoU calculation)
def fast_hist(target, prediction, num_classes):
    # Only consider valid pixels (ignore 255)
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + prediction[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

# 4. Preprocess and Postprocess (Exact match to your submission)
def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=769, interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    import torch.nn.functional as F
    pred_max = torch.argmax(pred, dim=1, keepdim=True).float()
    pred_resized = F.interpolate(pred_max, size=original_shape, mode='nearest')
    return pred_resized.cpu().detach().numpy().squeeze().astype(np.uint8)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    model = DeepLabV3Plus(Resnet_weights=False)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    # Load Validation Dataset (NO transforms here, we want the raw 1024x2048 images)
    val_dataset = Cityscapes(DATA_DIR, split="val", mode="fine", target_type="semantic")
    print(f"Evaluating on {len(val_dataset)} validation images...")

    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))

    with torch.no_grad():
        for i in tqdm(range(len(val_dataset))):
            img, target = val_dataset[i]
            original_shape = (target.height, target.width) # Should be 1024x2048

            # Process prediction
            img_tensor = preprocess(img).to(device)
            pred_logits = model(img_tensor)
            pred_mask = postprocess(pred_logits, original_shape)

            # Process ground truth
            target_mask = convert_target_to_trainid(target)

            # Update confusion matrix
            hist += fast_hist(target_mask, pred_mask, NUM_CLASSES)

    # Calculate final metrics
    ious = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(ious)

    print("\n" + "="*30)
    print(f"Final Mean IoU: {miou * 100:.2f}%")
    print("="*30)
    
    # Optional: Print per-class IoU
    class_names = [cls.name for cls in Cityscapes.classes if cls.train_id != 255 and cls.train_id != -1]
    for name, iou in zip(class_names, ious):
        print(f"{name:>15}: {iou * 100:.2f}%")

if __name__ == "__main__":
    main()