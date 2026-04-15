wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.002 \
    --num-workers 24 \
    --seed 42 \
    --experiment-id "UNet_CrossEntropy_64" \
    --loss-function "cross_entropy"\
    --model-arch "unet" \
