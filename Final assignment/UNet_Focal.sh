wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.004 \
    --num-workers 24 \
    --seed 42 \
    --experiment-id "UNet_Focal_64" \
    --loss-function "focal"\
    --model-arch "unet" \
