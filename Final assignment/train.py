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
    RandomCrop
)
import torchvision.transforms.v2.functional as TF
import torch.nn.functional as F

from model import Model, DeepLabV3Plus

import torch.nn.functional as F

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

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

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

    return parser

class JointTransform:
    def __init__(self, crop_size=513, is_train=True):
        self.crop_size = crop_size
        self.is_train = is_train

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
    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        # Use the plural 'transforms' argument!
        transforms=JointTransform(crop_size=769, is_train=True),
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        # Use center crop for validation
        transforms=JointTransform(crop_size=769, is_train=False),
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = DeepLabV3Plus(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)



    # Separate the parameters
    backbone_params = list(model.low_level_features.parameters()) + list(model.high_level_features.parameters())
    head_params = list(model.aspp.parameters()) + list(model.decoder.parameters())


    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)

    # Define the loss function
    #criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class
    criterion = FocalLoss(ignore_index=255, gamma=2.0)  # Ignore the void class
    
    def backbone_lambda(current_iter):
        current_epoch = current_iter // len(train_dataloader)
        if current_epoch < 5:
            return 0.0  # Keep frozen for first 5 epochs
        else:
            return (1 - current_iter / max_iters) ** 0.9  # Poly decay after epoch 5

    def head_lambda(current_iter):
            return (1 - current_iter / max_iters) ** 0.9 # Poly decay from day 1

    optimizer=SGD([
        {'params': backbone_params, 'lr': args.lr * 0.1}, # 10x smaller for the backbone
        {'params': head_params, 'lr': args.lr}            # Normal rate for the heads
    ], momentum=0.9, weight_decay=1e-4)
   #scheduler 
    max_iters = len(train_dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=[backbone_lambda, head_lambda]
    )

   
    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")
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
            scheduler.step()  # Step the scheduler

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[1]['lr'],
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
