wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 40 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "DeepLabV3+-v7-training" \