wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.002 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "UNet_Focal" \
    --loss-function "focal"\
    --model-arch "unet" \
