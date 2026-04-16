"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""

MODEL_PATH = "model.pt"
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
    RandomCrop,
    ColorJitter,
)
import torchvision.transforms.v2.functional as TF
import torch.nn.functional as F

from model import Model, DeepLabV3Plus

import torch.nn.functional as F

class CEDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, ce_weight=1.0, dice_weight=1.0, class_weights=None):
        """
        Combines Cross Entropy Loss and Dice Loss.
        ce_weight: The multiplier for the CE loss component.
        dice_weight: The multiplier for the Dice loss component.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # Standard Cross Entropy
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        
        # Small constant to prevent division by zero
        self.smooth = 1e-5 

    def forward(self, logits, targets):
        # 1. Calculate Standard Cross Entropy
        ce_loss = self.ce(logits, targets)

        # 2. Prepare for Dice Loss Calculation
        # Convert logits to probabilities (Softmax)
        probs = F.softmax(logits, dim=1)
        
        # Create a mask for valid pixels (ignore the 255s)
        valid_mask = (targets != self.ignore_index).unsqueeze(1) # Shape: [B, 1, H, W]
        
        # 3. One-Hot Encode the Targets Safely
        # PyTorch one_hot crashes if it sees index 255 for a 19-class model.
        # So we temporarily clone targets and replace 255 with 0.
        safe_targets = targets.clone()
        safe_targets[targets == self.ignore_index] = 0
        
        # Convert to one-hot: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
        targets_one_hot = F.one_hot(safe_targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        
        # 4. Apply the valid mask
        # This completely zeros out the probabilities and targets wherever the original label was 255
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        # 5. Calculate Dice Score
        # Sum over batch, height, and width (dims 0, 2, 3), leaving the Class dimension
        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))
        
        # Formula: (2 * Intersection) / (Prediction + Target)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # We want to MINIMIZE loss, so we return 1 - mean(dice_score)
        dice_loss = 1.0 - torch.mean(dice_score)
        
        # 6. Combine and return
        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, reduction='mean'):
        super().__init__()
        # alpha: Tensor of class weights (like the Cityscapes weights)
        # gamma: The "volume knob" to squash easy examples (default 2.0 is standard)
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Compute the log probabilities (log_pt) and probabilities (pt) for all classes
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)

        # 2. Calculate the focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # 3. Multiply the log probabilities by the focal weight
        weighted_log_pt = focal_weight * log_pt

        # 4. Use standard NLLLoss to extract the true target classes, 
        #    apply the alpha class weights, and ignore the void index (255)
        loss = F.nll_loss(
            weighted_log_pt, 
            targets, 
            weight=self.alpha, 
            ignore_index=self.ignore_index, 
            reduction=self.reduction
        )

        return loss

# Mapping class IDs to train IDs (Keep this line!)
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

# --- NEW VECTORIZED MAPPING ---
mapping_tensor = torch.zeros(256, dtype=torch.long)
for city_id, train_id in id_to_trainid.items():
    mapping_tensor[city_id] = train_id

def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    # Vectorized GPU/CPU compatible replacement!
    return mapping_tensor[label_img.long()]
# ------------------------------

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--loss-function", type=str, default="focal", choices=["focal", "cross_entropy", "cross_entropy_weighted", "ce_dice"], help="Select Focal Loss or Cross Entropy Loss")
    parser.add_argument("--model-arch", type=str, default="deeplabv3plus", choices=["deeplabv3plus", "unet"], help="Model architecture to use")
    parser.add_argument("--continue-from", type=str, default=None, help="Path to a checkpoint to continue training from")
    parser.add_argument("--resnet-size", type=int, default=101, choices=[50, 101], help="Size of ResNet backbone for DeepLabV3+ (50 or 101)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Whether to freeze the backbone of DeepLabV3+ during training")

    return parser

class JointTransform:
    def __init__(self, crop_size=513, is_train=True):
        self.crop_size = crop_size
        self.is_train = is_train
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __call__(self, image, target):
        # 1. Convert PIL images to tensors
        image = TF.to_image(image)
        target = TF.to_image(target)

        # 2. Resize maintaining aspect ratio! 
        # By passing a single int (self.crop_size) instead of a tuple (512, 512),
        # PyTorch resizes the shorter edge to 513. 1024x2048 becomes 513x1026.
        image = TF.resize(image, self.crop_size, interpolation=InterpolationMode.BILINEAR)
        target = TF.resize(target, self.crop_size, interpolation=InterpolationMode.NEAREST)

        # 3. Apply the Crop
        if self.is_train:
            # Generate random coordinates once...
            i, j, h, w = RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            # ...and apply them to BOTH image and mask
            image = TF.crop(image, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
            if torch.rand(1) < 0.5:  # Random horizontal flip with 50% chance
                image = TF.hflip(image)
                target = TF.hflip(target)
            image = self.color_jitter(image)
        else:
            # For validation, we just take the center to keep it consistent
            image = TF.center_crop(image, output_size=(self.crop_size, self.crop_size))
            target = TF.center_crop(target, output_size=(self.crop_size, self.crop_size))

        # 4. Final Formatting
        image = TF.to_dtype(image, torch.float32, scale=True)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        target = TF.to_dtype(target, torch.int64)

        return image, target

def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the dataset and make a split for training and validation
    if args.model_arch == "unet":
        train_dataset = Cityscapes(
            args.data_dir,
            split="train",
            mode="fine",
            target_type="semantic",
            # Use the plural 'transforms' argument!
            transforms=JointTransform(crop_size=512, is_train=True),
        )

        valid_dataset = Cityscapes(
            args.data_dir,
            split="val",
            mode="fine",
            target_type="semantic",
            # Use center crop for validation
            transforms=JointTransform(crop_size=512, is_train=False),
        )
    elif args.model_arch == "deeplabv3plus":
        train_dataset = Cityscapes(
            args.data_dir,
            split="train",
            mode="fine",
            target_type="semantic",
            # Use the plural 'transforms' argument!
            transforms=JointTransform(crop_size=513, is_train=True),
        )

        valid_dataset = Cityscapes(
            args.data_dir,
            split="val",
            mode="fine",
            target_type="semantic",
            # Use center crop for validation
            transforms=JointTransform(crop_size=513, is_train=False),
        )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,           # Transfers data to GPU faster
        persistent_workers=True,    # Keeps workers "warm" between epochs
        prefetch_factor=2          # Workers will prepare 2 batches in advance
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    if args.model_arch == "unet":
        model = Model()
    elif args.model_arch == "deeplabv3plus":
        if args.continue_from is None:
            model = DeepLabV3Plus(
                in_channels=3,  # RGB images
                n_classes=19,  # 19 classes in the Cityscapes dataset
                Resnet_weights=True,  # Use pretrained weights for the backbone
                Resnet_size=args.resnet_size,  # Use the specified ResNet size
            )
        else:
            model = DeepLabV3Plus(
                in_channels=3,  # RGB images
                n_classes=19,
                Resnet_weights=False,  # Don't load pretrained weights for the backbone
                Resnet_size=args.resnet_size,  # Use the specified ResNet size
            )
            state_dict = torch.load(
                args.continue_from, 
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(
                state_dict, 
                strict=True,  # Ensure the state dict matches the model architecture
            )

    model.to(device)

    # Define the loss function
    # 1. Define the smoothed heuristic weights for Cityscapes
    cityscapes_weights = torch.tensor([
        0.05, 0.05, 0.05, # 0, 1, 2: Road, Sidewalk, Building (Massive classes)
        0.20, 0.20, 0.20, # 3, 4, 5: Wall, Fence, Pole (Thin/Medium classes)
        0.80, 0.80,       # 6, 7: Traffic Light, Traffic Sign (Tiny classes)
        0.05, 0.10,       # 8, 9: Vegetation, Terrain
        0.05, 0.30, 0.50, # 10, 11, 12: Sky, Person, Rider
        0.10, 0.40, 0.80, # 13, 14, 15: Car, Truck, Bus
        0.80, 0.80, 0.80  # 16, 17, 18: Train, Motorcycle, Bicycle
    ], dtype=torch.float32).to(device)
    if args.loss_function == "cross_entropy_weighted":
        criterion = nn.CrossEntropyLoss(ignore_index=255, weight=cityscapes_weights)  # Ignore the void class
    elif args.loss_function == "focal":
        criterion = FocalLoss(ignore_index=255, gamma=2.0)  # Ignore the void class
    elif args.loss_function == "cross_entropy":
        criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class
    elif args.loss_function == "ce_dice":
        criterion = CEDiceLoss(ignore_index=255, ce_weight=1.0, dice_weight=1.0, class_weights=cityscapes_weights)  # Equal weighting for CE and Dice
    
    max_iters = len(train_dataloader) * args.epochs

    def backbone_lambda(current_iter):
        current_epoch = current_iter // len(train_dataloader)
        if args.model_arch == "deeplabv3plus" and args.freeze_backbone:
            if current_epoch < 10:
                return 0.0  # Keep frozen for first 10 epochs
        return (1 - current_iter / max_iters) ** 0.9  # Poly decay after epoch 10

    def head_lambda(current_iter):
            return (1 - current_iter / max_iters) ** 0.9 # Poly decay from day 1
    scheduler = None
    if args.model_arch == "unet":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.model_arch == "deeplabv3plus":
        # Separate backbone and head parameters
        backbone_params = list(model.low_level_features.parameters()) + list(model.high_level_features.parameters())
        head_params = list(model.aspp.parameters()) + list(model.decoder.parameters())
        backbone_lr = args.lr * 0.5 if args.freeze_backbone else args.lr * 0.1
        
        optimizer = SGD([
            {'params': backbone_params, 'lr': backbone_lr}, 
            {'params': head_params, 'lr': args.lr}          
        ], momentum=0.9, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=[backbone_lambda, head_lambda]
        )
   #scheduler 
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)
    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")
        
        # --- THE FREEZE/UNFREEZE LOGIC ---
        if args.model_arch == "deeplabv3plus" and args.freeze_backbone:
            if epoch < 10:
                # Freeze backbone
                for param in model.module.low_level_features.parameters() if hasattr(model, 'module') else model.low_level_features.parameters():
                    param.requires_grad = False
                for param in model.module.high_level_features.parameters() if hasattr(model, 'module') else model.high_level_features.parameters():
                    param.requires_grad = False
            elif epoch == 10:
                # Unfreeze backbone and update optimizer learning rate
                print("Unfreezing backbone!")
                for param in model.module.low_level_features.parameters() if hasattr(model, 'module') else model.low_level_features.parameters():
                    param.requires_grad = True
                for param in model.module.high_level_features.parameters() if hasattr(model, 'module') else model.high_level_features.parameters():
                    param.requires_grad = True
                    
        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # Step the scheduler

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[-1]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(
        model_to_save.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:.4}.pt" # Fixed string format!
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
